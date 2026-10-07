"""Portable external frozen inference using verified original definitions.

Only selected definitions are loaded from legacy drivers: their top-level
training/configuration/file operations are never executed. Function and class
bodies are compiled unchanged. External weights and inputs remain read-only.
"""
from __future__ import annotations

import ast
from pathlib import Path
import types

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

import reproduce as r


def definitions(relative, names, namespace, owner=None):
    path = r.local_path(r.EXPERIMENTS, relative)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if owner:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Missing original definition in {relative}: {names}")
    # Compile original syntax nodes, without rewriting any operation.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


def preprocessing():
    return definitions("external_frozen_phase_20261001/external_inputs.py",
                       ("normalize_native", "low_image"), {"np": np, "Image": Image})


def prepare_image(path, case):
    with Image.open(path) as handle:
        raw = np.array(handle)
    if raw.ndim != 2 or list(raw.shape) != case["raw_grid_yx"] or not np.isfinite(raw).all():
        raise ValueError(f"Expected finite original grayscale grid {case['raw_grid_yx']}; do not supply an already cropped/resized image")
    if case["crop_yx"] is not None:
        (y0, y1), (x0, x1) = case["crop_yx"]
        raw = raw[y0:y1, x0:x1]
    ops = preprocessing()
    native = ops.normalize_native(raw)
    if list(native.shape) != case["native_grid_yx"]:
        raise ValueError("Unexpected external native grid")
    return native, ops.low_image(native)


def verify_sources(catalog):
    for name, expected in catalog["sources"].items():
        if r.sha256(r.local_path(r.EXPERIMENTS, name)) != expected:
            raise ValueError(f"External source changed: {name}")


def checked_resource(root, entry):
    from shared_resources import checked
    return checked(root, entry)


class ExternalPipeline:
    def __init__(self, case, resource_root):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The archived inference requires CUDA/BF16")
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(0.35)
        core, _ = r.pipeline_modules()
        prior = r.module("pack_external_prior", "external_clean_prior_adaptation_20261001/prior_schema.py")
        if prior.PROMPTS[case["dataset"]] != case["description"]:
            raise ValueError("Prior differs from the completed external protocol")
        seed = case["seed"]
        main = r.read_json(r.PACKAGE / "main_checkpoints.json")[str(seed)]
        initial = {key: checked_resource(resource_root, main[key])
                   for key in ("tokenizer", "global_checkpoint", "native_checkpoint")}
        checkpoints = {key: checked_resource(resource_root, entry) for key, entry in case["weights"].items()}
        projection = definitions("external_frozen_phase_20261001/phase_predictor.py",
                                 ("project_native",), {"torch": torch, "F": F})
        context = types.SimpleNamespace(ARMS=("global", "native_detail"), SEED=seed,
            locked=types.SimpleNamespace(paths=lambda requested_seed: initial if requested_seed == seed else None),
            order=core.order, NativeNoContext=core.NativeNoContext, project_native=projection.project_native)
        adapter = definitions("external_clean_prior_adaptation_20261001/model_adapter.py",
            ("AdaptationModel", "frozen_from_state"), {"torch": torch, "nn": nn, "F": F, "c": context,
            "extend_tokenizer": prior.extend_tokenizer, "AppendOnlyEmbedding": prior.AppendOnlyEmbedding})
        self.global_model = adapter.AdaptationModel("global")
        self.native_model = adapter.AdaptationModel("native_detail")
        self.head = core.BodyInterfaceDecoder("separate")
        for role, model in (("global", self.global_model), ("native_detail", self.native_model), ("body", self.head)):
            saved = torch.load(checkpoints[role], map_location="cpu", weights_only=False)
            entry = case["weights"][role]
            if saved["epoch"] != entry["epoch"] or saved["dataset"] != case["dataset"] or saved["arm"] != entry["arm"]:
                raise ValueError(f"Wrong checkpoint role or selection: {role}")
            model.load_state_dict(saved["model"], strict=True)
            del saved
            model.eval().requires_grad_(False).cuda()
        self.global_model.to(memory_format=torch.channels_last)
        self.native_model.to(memory_format=torch.channels_last)
        self.features = None
        self.hook = self.native_model.native_model.up_fine.register_forward_hook(
            lambda module, args, output: setattr(self, "features", output))
        self.description = case["description"]
        self.g_ids, self.g_attention = self.global_model.tokenizer.encode(self.description)
        self.n_ids, self.n_attention = self.native_model.tokenizer.encode(self.description)
        self.g_ids, self.g_attention = self.g_ids.cuda()[None], self.g_attention.cuda()[None]
        self.n_ids, self.n_attention = self.n_ids.cuda()[None], self.n_attention.cuda()[None]
        spatial = definitions("external_body_adaptation_20261003/spatial.py", ("nearest512_ids",), {"np": np, "Image": Image})
        if case["descriptor_support"] == "logit_argmax":
            rep = r.module("pack_external_representation", "external_fine8_multiseed_body_recovery_20261005/representation_authoritative.py")
            predictor = "external_fine8_multiseed_test_recovery_20261005/evaluate_multiseed.py"
            describe = rep.describe
        elif case["descriptor_support"] == "probability_argmax":
            predictor = "external_full_vs_global_fine8_test_20261004/evaluate_locked.py"
            describe = core.describe
        else:
            raise ValueError("Unknown descriptor support protocol")
        functions = definitions(predictor, ("predict",), {"np": np, "torch": torch,
            "c": types.SimpleNamespace(describe=describe, assign_bodies=core.assign_bodies),
            "s": spatial, "old": types.SimpleNamespace(s=spatial)}, owner="Pipeline")
        self.predict = types.MethodType(functions.predict, self)

    def close(self):
        self.hook.remove()
        self.features = None
        for model in (self.global_model, self.native_model, self.head):
            model.cpu()
        torch.cuda.empty_cache()


def run(args):
    if not args.trusted_checkpoints:
        raise ValueError("Only verified, trusted local checkpoints may be loaded")
    r.verify_contents()
    out = r.new_output(args.output)
    catalog = r.read_json(r.PACKAGE / "external_checkpoints.json")
    verify_sources(catalog)
    if args.case not in catalog["cases"]:
        raise ValueError(f"Choose an explicit archived case: {list(catalog['cases'])}")
    case = catalog["cases"][args.case]
    image, low = prepare_image(args.image, case)
    pipeline = ExternalPipeline(case, args.resource_root)
    try:
        predictions = pipeline.predict(image, low)
    finally:
        pipeline.close()
    upstream, complete = predictions["native_upstream"], predictions["native_complete"]
    np.testing.assert_array_equal(upstream >= 3, complete >= 3)
    np.testing.assert_array_equal(upstream[upstream < 3], complete[upstream < 3])
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, values in predictions.items():
        if values.dtype != np.uint8 or values.ndim != 2 or values.max() > 7:
            raise ValueError(f"Unexpected prediction {name}")
        path = out / f"{name}.png"
        Image.fromarray(values).save(path)
        files[name] = {"shape": list(values.shape), "sha256": r.sha256(path)}
    report = {"case": args.case, "seed": case["seed"], "dataset": case["dataset"], "attempt": case["attempt"],
              "selected_epochs": {role: entry["epoch"] for role, entry in case["weights"].items()},
              "image_sha256": r.sha256(args.image), "description": case["description"],
              "normalized_image_sha256": __import__("hashlib").sha256(image.tobytes()).hexdigest(),
              "low_image_sha256": __import__("hashlib").sha256(low.tobytes()).hexdigest(),
              "batch_size": 1, "precision": catalog["precision"], "descriptor_support": case["descriptor_support"],
              "oil_support_and_non_oil_labels_preserved": True, "reference_labels_read": False,
              "training_started": False, "outputs": files}
    r.write_json(out / "prediction.json", report)
    return {"status": "PASS", "case": args.case, "output": str(out), "maps": len(files)}
