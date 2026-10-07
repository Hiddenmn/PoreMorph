# -*- coding: utf-8 -*-
"""Matched fusion-order experiment for the Neurocomputing manuscript.

The reference model applies PPM before cross-attention. This script keeps every
module and training setting from run_seed_statistics.py but swaps those two
operations: cross-attention first, then the same PPM. The image-only lateral
paths are unchanged. Run from the project root or any other working directory:

    python MM-PoreFormer/scripts/run_fusion_order_ablation.py

Checkpoints are isolated from all existing submission and experiment files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import run_seed_statistics as base


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "实验"
OUTPUT_ROOT = DATA_ROOT / "multimodal_remaining_oil_outputs_fusion_order_ablation"
BASELINE_ROOT = DATA_ROOT / "multimodal_remaining_oil_outputs_repeat_seed_mppvit_vs_transunet"
MAIN_BASELINE = DATA_ROOT / "multimodal_remaining_oil_outputs_fast_ppm_before_fusion"
ORIGINAL_BUILD_MODEL = base.build_model


class FusionBeforePPM(base.MultiModalViTPPMTransformerUNet):
    """Identical modules to the full model, with only the fusion order swapped."""

    def forward(self, image, input_ids=None, attention_mask=None):
        feat, vit_feats, skips = self.visual_encoder(image)
        s1, s2, s3, s4 = skips
        v1, v2, v3 = vit_feats

        if self.use_text and input_ids is not None and attention_mask is not None:
            text_feats = self.text_encoder(input_ids, attention_mask)
            batch, channels, height, width = feat.shape
            visual_tokens = feat.flatten(2).transpose(1, 2)
            fused_tokens, _ = self.cross_attn(
                query=self.cross_norm(visual_tokens),
                key=text_feats,
                value=text_feats,
                key_padding_mask=attention_mask == 0,
                need_weights=False,
            )
            visual_tokens = visual_tokens + self.text_alpha * fused_tokens
            feat = visual_tokens.transpose(1, 2).reshape(batch, channels, height, width)

        if self.use_ppm:
            feat = feat + base.Config.ppm_scale * self.ppm(feat)

        v_s2 = F.interpolate(self.vit_to_s2(v1), size=s2.shape[-2:], mode="bilinear", align_corners=False)
        v_s3 = F.interpolate(self.vit_to_s3(v2), size=s3.shape[-2:], mode="bilinear", align_corners=False)
        v_s4 = F.interpolate(self.vit_to_s4(v3), size=s4.shape[-2:], mode="bilinear", align_corners=False)
        s2 = s2 + self.alpha_s2 * v_s2
        s3 = s3 + self.alpha_s3 * v_s3
        s4 = s4 + self.alpha_s4 * v_s4

        x = self.bridge(feat)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.final_conv(x)


def build_model(model_type: str, vocab_size: int):
    if model_type != "fusion_before_ppm":
        return ORIGINAL_BUILD_MODEL(model_type, vocab_size)
    return FusionBeforePPM(
        vocab_size=vocab_size,
        num_classes=base.Config.num_classes,
        img_size=base.Config.target_size[0],
        patch_size=base.Config.patch_size,
        use_text=True,
        use_ppm=True,
        use_ppm_before_fusion=False,
    )


def baseline_summary(seed: int) -> dict:
    if seed == 42:
        path = MAIN_BASELINE / "results" / "01_PPM_before_Fusion_full" / "summary.json"
    else:
        path = BASELINE_ROOT / "results" / f"seed_{seed}_01_MPP_ViT_full" / "summary.json"
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("model_type") != "ppm_before_fusion_full":
        raise ValueError(f"Unexpected baseline model in {path}")
    return {"path": str(path), **summary}


def restore_original_conditioning(records: list[dict]) -> dict[str, set[tuple[str, int]]]:
    """Use the exact input descriptions saved with the original seed-42 run."""
    split_dir = MAIN_BASELINE / "results" / "dataset_splits"
    descriptions: dict[tuple[str, int], str] = {}
    split_keys: dict[str, set[tuple[str, int]]] = {}
    for split in ("train", "val", "test"):
        with (split_dir / f"{split}_records.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys = {(row["sample"], int(row["slice_id"])) for row in rows}
        if len(keys) != len(rows):
            raise ValueError(f"Duplicate keys in saved {split} split")
        split_keys[split] = keys
        for row in rows:
            key = row["sample"], int(row["slice_id"])
            if key in descriptions:
                raise ValueError(f"Overlapping saved splits at {key}")
            descriptions[key] = row["text"]
    current_keys = {(row["sample"], int(row["slice_id"])) for row in records}
    if current_keys != set(descriptions) or len(current_keys) != len(records):
        raise ValueError("Current images/masks do not match the saved original input records")
    for record in records:
        record["text"] = descriptions[(record["sample"], int(record["slice_id"]))]
    return split_keys


def check_original_tokenizer(seed: int, tokenizer: base.CharTokenizer) -> None:
    if seed == 42:
        path = MAIN_BASELINE / "models" / "01_PPM_before_Fusion_full" / "tokenizer.json"
    else:
        path = BASELINE_ROOT / "models" / f"seed_{seed}_01_MPP_ViT_full" / "tokenizer.json"
    with path.open(encoding="utf-8") as stream:
        original = json.load(stream)
    if original["idx2token"] != tokenizer.idx2token:
        raise ValueError(f"Tokenizer mismatch against {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 2025])
    parser.add_argument("--smoke", action="store_true", help="Use a separate output directory and one epoch")
    args = parser.parse_args()

    # Windows DataLoader workers import the base module afresh. Its default
    # relative cache path must resolve to the same directory in every worker.
    os.chdir(DATA_ROOT)
    base.Config.project_root = DATA_ROOT
    base.Config.cache_dir = MAIN_BASELINE / "preprocessed_cache_512"
    base.Config.output_dir = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_smoke") if args.smoke else OUTPUT_ROOT
    base.Config.model_dir = base.Config.output_dir / "models"
    base.Config.result_dir = base.Config.output_dir / "results"
    base.Config.pred_dir = base.Config.output_dir / "predictions"
    if args.smoke:
        base.Config.num_epochs = 1
        base.Config.patience = 1
    base.build_model = build_model

    base.ensure_dirs()
    device = base.get_device()
    records = base.collect_records()
    if base.Config.use_preprocessed_cache:
        records = base.build_preprocessed_cache(records)
    saved_split_keys = restore_original_conditioning(records)

    results = []
    for seed in args.seeds:
        base.Config.seed = seed
        base.set_seed(seed)
        train_records, val_records, test_records = base.split_records_by_sample(records)
        if seed == 42:
            for split, rows in (("train", train_records), ("val", val_records), ("test", test_records)):
                keys = {(row["sample"], int(row["slice_id"])) for row in rows}
                if keys != saved_split_keys[split]:
                    raise ValueError(f"Seed-42 {split} split differs from the original experiment")
        tokenizer = base.CharTokenizer([record["text"] for record in train_records], max_len=base.Config.max_text_len)
        check_original_tokenizer(seed, tokenizer)
        train_dataset = base.MultimodalRemainingOilDataset(train_records, tokenizer, base.Config.target_size, augment=True)
        val_dataset = base.MultimodalRemainingOilDataset(val_records, tokenizer, base.Config.target_size, augment=False)
        test_dataset = base.MultimodalRemainingOilDataset(test_records, tokenizer, base.Config.target_size, augment=False)
        train_loader = base.build_loader(train_dataset, base.Config.batch_size, shuffle=True, drop_last=True)
        val_loader = base.build_loader(val_dataset, base.Config.batch_size, shuffle=False, drop_last=False)
        test_loader = base.build_loader(test_dataset, base.Config.batch_size, shuffle=False, drop_last=False)
        class_weights = base.compute_auto_class_weights(train_records, base.Config.class_weight_max_samples)

        experiment = f"seed_{seed}_fusion_before_ppm"
        summary = base.train_experiment(
            exp_cfg={"name": experiment, "model_type": "fusion_before_ppm"},
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            device=device,
        )
        original = baseline_summary(seed)
        summary.update({
            "seed": seed,
            "fusion_order": "ViT -> Cross-Attention -> Plain PPM -> Decoder",
            "baseline": original,
            "test_mIoU_difference_baseline_minus_variant": original["test_mIoU"] - summary["test_mIoU"],
            "matched_protocol": {
                "split_mode": base.Config.split_mode,
                "num_epochs": base.Config.num_epochs,
                "batch_size": base.Config.batch_size,
                "learning_rate": base.Config.learning_rate,
                "weight_decay": base.Config.weight_decay,
                "text_dropout_prob": base.Config.text_dropout_prob,
                "cache_dir": str(base.Config.cache_dir),
                "original_conditioning_source": str(MAIN_BASELINE / "results" / "dataset_splits"),
                "train_count": len(train_records),
                "val_count": len(val_records),
                "test_count": len(test_records),
            },
        })
        result_file = base.Config.result_dir / experiment / "summary.json"
        with result_file.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        results.append(summary)
        torch.cuda.empty_cache()

    all_results = base.Config.result_dir / "fusion_order_comparison.json"
    with all_results.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    for result in results:
        print(
            f"seed={result['seed']}: original={result['baseline']['test_mIoU']:.4f}, "
            f"fusion-before-PPM={result['test_mIoU']:.4f}, "
            f"difference={result['test_mIoU_difference_baseline_minus_variant']:+.4f}"
        )
    print(f"Saved comparison: {all_results}")


if __name__ == "__main__":
    main()
