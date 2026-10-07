# -*- coding: utf-8 -*-
"""
MPP-ViT vs TransUNet 多随机种子重复实验与统计性检验脚本
============================================================

适用数据：
- Oil
- NPI-006
- NPI-012
- WI

功能：
1. 自动读取四类样本的原始图像、语义分割图、jsonl文本模态；
2. 标签标准固定为：
   0 背景
   1 岩石骨架
   2 孔隙水
   3 膜状油
   4 孔隙角油
   5 滴状油
   6 柱状油
   7 簇状油
3. 每种类型内部按 7:2:1 划分 train/val/test；
4. 先训练论文主模型 MPP-ViT，再进行测试评估；
5. 然后依次训练对比实验和消融实验模型，并完成测试评估；
6. 默认使用 GPU 训练，适配 RTX 5070 / CUDA 环境；
7. 输出训练曲线、best model、final model、测试集指标、类别级指标和可视化结果。

运行方式：
    python train_multimodal_remaining_oil.py

建议：
- 第一次跑可以把 Config.num_epochs 改小，比如 2 或 5，检查数据路径是否正确；
- 正式训练再改回 60 或更高；
- 如果显存不足，优先把 batch_size 从 4 改回 2；
- 第一次运行会构建预处理缓存，后续会明显更快；
- 如果 Windows 多进程 DataLoader 出问题，把 num_workers 改成 0 或 4。
"""

import os
import re
import json
import csv
import math
import time
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ============================================================
# 0. 全局配置
# ============================================================

@dataclass
class SampleConfig:
    key: str
    display_name: str
    original_dir_candidates: List[str]
    mask_dir_name: str
    text_dir_name: str
    image_prefix: str
    mask_prefix: str


class Config:
    # --------------------------------------------------------
    # 路径设置
    # --------------------------------------------------------
    project_root = Path(".").resolve()

    # 原始图像文件夹：你说“原始图片的文件夹和命名没有变化”，
    # 所以这里给出候选名；若候选名找不到，代码会根据 image_prefix 自动搜索。
    sample_configs = {
        "NPI-006": SampleConfig(
            key="NPI-006",
            display_name="NPI-006",
            original_dir_candidates=[
                "NPI-006过滤", "npi-006过滤", "NPI-006原始图", "npi-006原始图",
                "NPI-006", "npi-006"
            ],
            mask_dir_name="npi-006语义分割图",
            text_dir_name="npi-006文本",
            image_prefix="NPI-006-Reconstructed0000.filtered",
            mask_prefix="NPI-006-Reconstructed0000.filtered",
        ),
        "NPI-012": SampleConfig(
            key="NPI-012",
            display_name="NPI-012",
            original_dir_candidates=[
                "NPI-012过滤", "npi-012过滤", "NPI-012原始图", "npi-012原始图",
                "NPI-012", "npi-012"
            ],
            mask_dir_name="npi-012语义分割图",
            text_dir_name="npi-012文本",
            image_prefix="NPI-012-Reconstructed0000.filtered",
            mask_prefix="NPI-012-Reconstructed0000.filtered",
        ),
        "Oil": SampleConfig(
            key="Oil",
            display_name="Oil",
            original_dir_candidates=[
                "Oil过滤", "oil过滤", "Oil原始图", "oil原始图",
                "Oil", "oil"
            ],
            mask_dir_name="oil语义分割图",
            text_dir_name="oil文本",
            image_prefix="Oil-Reconstructed.filtered",
            mask_prefix="Oil-Reconstructed.filtered",
        ),
        "WI": SampleConfig(
            key="WI",
            display_name="WI",
            original_dir_candidates=[
                "WI过滤", "wi过滤", "WI原始图", "wi原始图",
                "WI", "wi"
            ],
            mask_dir_name="wi语义分割图",
            text_dir_name="wi文本",
            image_prefix="WI-Reconstructed0000.filtered",
            mask_prefix="WI-Reconstructed0000.filtered",
        ),
    }

    # 文本字段：建议使用刚生成的 text_for_model
    text_column = "text_for_model"

    # --------------------------------------------------------
    # 标签标准：必须与本次标注规则一致
    # --------------------------------------------------------
    num_classes = 8
    class_names = {
        0: "背景",
        1: "岩石骨架",
        2: "孔隙水",
        3: "膜状油",
        4: "孔隙角油",
        5: "滴状油",
        6: "柱状油",
        7: "簇状油",
    }

    # palette TIFF 如果被读成 RGB，可用这里映射回类别
    palette_rgb_to_class = {
        (0, 0, 0): 0,          # 背景
        (150, 150, 150): 1,    # 岩石骨架
        (0, 102, 255): 2,      # 孔隙水
        (255, 230, 0): 3,      # 膜状油
        (255, 140, 0): 4,      # 孔隙角油
        (140, 216, 255): 5,    # 滴状油
        (255, 0, 0): 6,        # 柱状油
        (160, 32, 240): 7,     # 簇状油
    }

    # --------------------------------------------------------
    # 数据划分
    # --------------------------------------------------------
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # random_per_sample：每一类样本内随机 7:2:1
    # block_per_sample：每一类样本内按切片顺序前70%/中20%/后10%
    split_mode = "block_per_sample"

    # --------------------------------------------------------
    # 图像与训练参数
    # --------------------------------------------------------
    target_size = (512, 512)
    in_channels = 3

    batch_size = 16
    num_workers = min(8, max(2, (os.cpu_count() or 8) // 2))
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 60
    patience = 10
    grad_accum_steps = 1
    clip_grad_norm = 1.0

    # RTX 5070 建议打开 AMP
    use_amp = True
    require_cuda = True

    # --------------------------------------------------------
    # 高速优化开关
    # --------------------------------------------------------
    # 1) 预处理缓存：第一次运行会把 TIFF + mask resize 后写入 memmap；
    #    后续训练直接读取缓存，能明显减少 CPU / PIL / TIFF 开销。
    use_preprocessed_cache = True
    force_rebuild_cache = False
    cache_dir = Path("multimodal_remaining_oil_outputs_fast_ppm_before_fusion") / "preprocessed_cache_512"

    # 2) DataLoader：Windows 下 workers 太大可能反而慢，建议 4/8/12 实测；
    #    这里默认最多 8，通常比 4 更快。
    prefetch_factor = 4

    # 3) channels_last 对卷积部分通常有提速，ViT 部分影响不大但一般安全。
    use_channels_last = True

    # 4) torch.compile：Windows 上可能不稳定，默认关闭；
    #    如果你的环境支持，可以改 True，第一次 epoch 会慢，后面可能加速。
    use_torch_compile = False
    torch_compile_mode = "max-autotune"

    # 5) 类别权重如果已有缓存，可直接从缓存 mask 统计，避免重复读 TIFF。
    use_cache_for_class_weights = True

    # 字符级 tokenizer；如果显存允许，text_for_model 建议 max_text_len 不要太短
    max_text_len = 192

    # ViT 主干
    patch_size = 16
    embed_dim = 192
    depth = 4
    num_heads = 6
    mlp_ratio = 4.0
    drop_rate = 0.1

    # 文本 Transformer
    text_embed_dim = 192
    text_depth = 2
    text_num_heads = 6
    text_drop_rate = 0.1
    text_alpha_init = 0.03
    text_dropout_prob = 0.30

    # PPM
    # 本版使用最开始的普通 PPM，不使用 TG-SPPM，也不使用文本引导尺度选择。
    # 顺序：ViT -> 普通PPM -> Cross-Attention图文融合 -> U-Net Decoder。
    ppm_scale = 0.15

    # 类别权重
    # auto：从训练集抽样统计类别频次；manual：使用下面 manual_class_weights
    class_weight_strategy = "auto"
    class_weight_max_samples = 600
    manual_class_weights = [0.4, 1.0, 1.2, 2.0, 2.2, 3.0, 2.4, 1.4]

    # 输出路径
    output_dir = Path("multimodal_remaining_oil_outputs_repeat_seed_mppvit_vs_transunet")
    model_dir = output_dir / "models"
    result_dir = output_dir / "results"
    pred_dir = output_dir / "predictions"

    # --------------------------------------------------------
    # 断点续训 / 安全暂停
    # --------------------------------------------------------
    auto_resume = True
    skip_completed_experiments = True
    save_resume_every_epoch = True
    stop_file_name = "STOP_TRAINING.txt"
    resume_checkpoint_name = "resume_checkpoint.pth"

    # 实验控制：论文主模型一定最先跑；后续为对比和消融
    run_baselines_and_ablations = True

    experiments = [
        # 论文主模型：普通PPM，且顺序为 ViT -> PPM -> Cross-Attention -> U-Net
        {"name": "01_PPM_before_Fusion_full", "model_type": "ppm_before_fusion_full"},

        # 关键消融：ViT + Text，不加PPM
        {"name": "02_Ablation_no_ppm", "model_type": "mpp_vit_no_ppm"},

        # 关键消融：ViT + 普通PPM，无文本
        {"name": "03_Ablation_no_text_ppm_before_fusion", "model_type": "ppm_before_fusion_no_text"},

        # 关键消融：纯ViT + U-Net
        {"name": "04_Ablation_no_text_no_ppm", "model_type": "mpp_vit_no_text_no_ppm"},

        # 对比实验：传统 U-Net
        {"name": "05_UNet_image_only", "model_type": "unet"},

        # 对比实验：CNN + PPM + U-Net
        {"name": "06_CNN_PPM_UNet_image_only", "model_type": "cnn_ppm_unet"},

        # 对比实验：TransUNet 风格图像单模态
        {"name": "07_TransUNet_image_only", "model_type": "transunet"},
    ]

    seed = 42
    repeat_seeds = [42, 2024, 2025]
    reuse_existing_seed42_results = True


# ============================================================
# 1. 工具函数
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def extract_slice_id(name: str) -> Optional[int]:
    m = re.search(r"filtered(\d+)", Path(name).name)
    if m:
        return int(m.group(1))
    return None


def ensure_dirs():
    for d in [Config.output_dir, Config.model_dir, Config.result_dir, Config.pred_dir]:
        d.mkdir(parents=True, exist_ok=True)


def get_device():
    if Config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("当前未检测到 CUDA。你要求在 GPU 上训练，请先确认 PyTorch CUDA 版本和显卡驱动。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("=" * 70)
        print("使用 GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        print("PyTorch version:", torch.__version__)
        print("=" * 70)

        # cuDNN/TF32/MatMul 优化。RTX 50 系一般能从这些设置中受益。
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        print(f"AMP dtype: {autocast_dtype()}")
        print(f"batch_size={Config.batch_size}, num_workers={Config.num_workers}, "
              f"prefetch_factor={Config.prefetch_factor}, channels_last={Config.use_channels_last}")
        print(f"preprocessed_cache={Config.use_preprocessed_cache}, torch_compile={Config.use_torch_compile}")
        print("=" * 70)
    else:
        print("警告：当前使用 CPU，训练会非常慢。")
    return device


def autocast_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    # RTX 40/50 系通常支持 bf16；若不支持则使用 fp16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def find_dir_case_insensitive(root: Path, dir_name: str) -> Optional[Path]:
    target = dir_name.lower()
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() == target:
            return p
    return None


def resolve_original_dir(cfg: SampleConfig) -> Path:
    root = Config.project_root

    for cand in cfg.original_dir_candidates:
        p = root / cand
        if p.exists() and p.is_dir():
            return p

    # 如果候选名找不到，根据原始图像前缀自动搜索
    forbidden = [
        cfg.mask_dir_name.lower(),
        cfg.text_dir_name.lower(),
        "语义分割图",
        "文本",
        "outputs",
        "models",
        "results",
    ]

    groups: Dict[Path, int] = {}
    for p in root.rglob(f"{cfg.image_prefix}*.tif"):
        low_parts = " ".join(part.lower() for part in p.parts)
        if any(k.lower() in low_parts for k in forbidden):
            continue
        groups[p.parent] = groups.get(p.parent, 0) + 1

    if not groups:
        raise FileNotFoundError(
            f"找不到 {cfg.display_name} 原始图像文件夹。"
            f"请检查候选文件夹 {cfg.original_dir_candidates}，或保证存在 {cfg.image_prefix}*.tif"
        )

    best = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    print(f"[自动匹配] {cfg.display_name} 原始图像文件夹：{best}，匹配 {groups[best]} 张")
    return best


def resolve_exact_dir(dir_name: str) -> Path:
    root = Config.project_root
    p = root / dir_name
    if p.exists() and p.is_dir():
        return p
    found = find_dir_case_insensitive(root, dir_name)
    if found is not None:
        return found
    raise FileNotFoundError(f"找不到文件夹：{dir_name}")


def read_jsonl_texts(text_dir: Path, text_column: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    返回：
    - filename_to_text
    - slice_id_to_text
    """
    jsonl_files = sorted(text_dir.glob("*.jsonl"), key=lambda p: natural_key(p.name))
    if not jsonl_files:
        raise FileNotFoundError(f"{text_dir} 中没有 jsonl 文本文件")

    filename_to_text: Dict[str, str] = {}
    sid_to_text: Dict[str, str] = {}

    for jf in jsonl_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                img_name = str(obj.get("image_filename", "")).strip()
                sid = str(obj.get("slice_id", "")).strip()

                txt = obj.get(text_column)
                if txt is None:
                    # 兜底字段
                    txt = obj.get("text_for_model") or obj.get("text_minimal") or obj.get("text_rich") or ""

                txt = str(txt)

                if img_name:
                    filename_to_text[Path(img_name).name] = txt
                if sid:
                    sid_to_text[sid.zfill(3)] = txt

    return filename_to_text, sid_to_text


def build_mask_map(mask_dir: Path) -> Dict[int, Path]:
    paths = sorted(
        [p for p in mask_dir.rglob("*") if p.suffix.lower() in [".tif", ".tiff"]],
        key=lambda p: natural_key(p.name)
    )
    mask_map: Dict[int, Path] = {}
    for p in paths:
        sid = extract_slice_id(p.name)
        if sid is not None:
            mask_map[sid] = p
    return mask_map


def collect_records() -> List[Dict[str, Any]]:
    """
    自动收集四类样本的 image-mask-text 三元组。
    """
    all_records = []

    print("\n开始扫描数据集...")
    for sample_key, cfg in Config.sample_configs.items():
        original_dir = resolve_original_dir(cfg)
        mask_dir = resolve_exact_dir(cfg.mask_dir_name)
        text_dir = resolve_exact_dir(cfg.text_dir_name)

        filename_to_text, sid_to_text = read_jsonl_texts(text_dir, Config.text_column)
        mask_map = build_mask_map(mask_dir)

        image_paths = sorted(
            [p for p in original_dir.rglob("*") if p.suffix.lower() in [".tif", ".tiff"] and p.name.startswith(cfg.image_prefix)],
            key=lambda p: natural_key(p.name)
        )

        if not image_paths:
            raise FileNotFoundError(f"{original_dir} 中未找到 {cfg.image_prefix}*.tif")

        sample_records = []
        missing_mask = []
        missing_text = []

        for img_path in image_paths:
            sid = extract_slice_id(img_path.name)
            if sid is None:
                continue

            mask_path = mask_map.get(sid)
            if mask_path is None:
                missing_mask.append(img_path.name)
                continue

            text = filename_to_text.get(img_path.name)
            if text is None:
                text = sid_to_text.get(f"{sid:03d}")

            if text is None:
                missing_text.append(img_path.name)
                continue

            sample_records.append({
                "sample": sample_key,
                "slice_id": sid,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "image_filename": img_path.name,
                "mask_filename": mask_path.name,
                "text": text,
            })

        sample_records = sorted(sample_records, key=lambda r: r["slice_id"])
        all_records.extend(sample_records)

        print("-" * 70)
        print(f"{sample_key}")
        print(f"  原始图像文件夹: {original_dir}")
        print(f"  语义分割文件夹: {mask_dir}")
        print(f"  文本文件夹: {text_dir}")
        print(f"  原始图像数量: {len(image_paths)}")
        print(f"  mask数量: {len(mask_map)}")
        print(f"  有效三元组: {len(sample_records)}")
        if sample_records:
            print(f"  范围: filtered{sample_records[0]['slice_id']:03d} - filtered{sample_records[-1]['slice_id']:03d}")
        if missing_mask:
            print(f"  缺少mask数量: {len(missing_mask)}，示例: {missing_mask[:3]}")
        if missing_text:
            print(f"  缺少文本数量: {len(missing_text)}，示例: {missing_text[:3]}")

    if not all_records:
        raise RuntimeError("没有收集到任何有效样本，请检查路径和命名。")

    print("=" * 70)
    print(f"总有效样本数: {len(all_records)}")
    print("=" * 70)

    return all_records


def split_records_by_sample(records: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(Config.seed)

    train_records, val_records, test_records = [], [], []

    for sample_key in Config.sample_configs.keys():
        group = [r for r in records if r["sample"] == sample_key]
        group = sorted(group, key=lambda r: r["slice_id"])

        if Config.split_mode == "random_per_sample":
            group = group.copy()
            rng.shuffle(group)
        elif Config.split_mode == "block_per_sample":
            pass
        else:
            raise ValueError(f"未知 split_mode: {Config.split_mode}")

        n = len(group)
        n_train = int(n * Config.train_ratio)
        n_val = int(n * Config.val_ratio)
        n_test = n - n_train - n_val

        train_records.extend(group[:n_train])
        val_records.extend(group[n_train:n_train + n_val])
        test_records.extend(group[n_train + n_val:])

        print(f"{sample_key} 划分：train={n_train}, val={n_val}, test={n_test}")

    # 训练集整体打乱；验证测试按样本和编号排序，方便看结果
    rng.shuffle(train_records)
    val_records = sorted(val_records, key=lambda r: (r["sample"], r["slice_id"]))
    test_records = sorted(test_records, key=lambda r: (r["sample"], r["slice_id"]))

    return train_records, val_records, test_records


# ============================================================
# 2. Tokenizer
# ============================================================

class CharTokenizer:
    def __init__(self, texts: List[str], max_len: int = 192):
        self.max_len = max_len
        vocab = set()
        for t in texts:
            for ch in str(t):
                vocab.add(ch)

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.cls_token = "<CLS>"

        self.idx2token = [self.pad_token, self.unk_token, self.cls_token] + sorted(list(vocab))
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}

        self.pad_id = self.token2idx[self.pad_token]
        self.unk_id = self.token2idx[self.unk_token]
        self.cls_id = self.token2idx[self.cls_token]

    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        text = str(text)
        ids = [self.cls_id]
        for ch in text:
            ids.append(self.token2idx.get(ch, self.unk_id))

        ids = ids[:self.max_len]
        attn_mask = [1] * len(ids)

        while len(ids) < self.max_len:
            ids.append(self.pad_id)
            attn_mask.append(0)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.idx2token)

    def save(self, path: Path):
        obj = {
            "max_len": self.max_len,
            "idx2token": self.idx2token,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls([], max_len=obj["max_len"])
        tok.idx2token = obj["idx2token"]
        tok.token2idx = {t: i for i, t in enumerate(tok.idx2token)}
        tok.pad_id = tok.token2idx[tok.pad_token]
        tok.unk_id = tok.token2idx[tok.unk_token]
        tok.cls_id = tok.token2idx[tok.cls_token]
        return tok


# ============================================================
# 3. Dataset
# ============================================================

def read_image_gray(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def read_mask_label(path: str) -> Image.Image:
    """
    读取语义分割图。
    如果是 palette TIFF，np.array 直接得到类别ID。
    如果被保存成 RGB，则按 Config.palette_rgb_to_class 映射。
    """
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        # 若原本就是 0-7 标签，直接返回
        if arr.max() <= Config.num_classes - 1:
            return Image.fromarray(arr.astype(np.uint8), mode="L")

        # 如果是灰度色值但不是标签，尽量兜底：clip
        arr = np.clip(arr, 0, Config.num_classes - 1).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    if arr.ndim == 3:
        out = np.zeros(arr.shape[:2], dtype=np.uint8)
        rgb = arr[:, :, :3]
        for color, cls in Config.palette_rgb_to_class.items():
            mask = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
            out[mask] = cls
        return Image.fromarray(out, mode="L")

    raise ValueError(f"无法识别mask维度: {path}, shape={arr.shape}")



# ============================================================
# 3.1 高速预处理缓存：TIFF/mask -> uint8 memmap
# ============================================================

def make_record_key(record: Dict[str, Any]) -> str:
    """用于判断缓存是否与当前数据集一致。"""
    return f"{record['sample']}|{record['slice_id']:03d}|{Path(record['image_path']).resolve()}|{Path(record['mask_path']).resolve()}"


def get_cache_paths():
    cache_dir = Config.cache_dir
    return {
        "dir": cache_dir,
        "meta": cache_dir / "cache_meta.json",
        "images": cache_dir / "images_uint8.dat",
        "masks": cache_dir / "masks_uint8.dat",
    }


def cache_is_valid(records: List[Dict[str, Any]]) -> bool:
    paths = get_cache_paths()
    if Config.force_rebuild_cache:
        return False
    if not paths["meta"].exists() or not paths["images"].exists() or not paths["masks"].exists():
        return False

    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        h, w = Config.target_size[1], Config.target_size[0]
        if meta.get("num_records") != len(records):
            return False
        if tuple(meta.get("target_size", [])) != tuple(Config.target_size):
            return False
        if meta.get("height") != h or meta.get("width") != w:
            return False
        old_keys = meta.get("record_keys", [])
        new_keys = [make_record_key(r) for r in records]
        if old_keys != new_keys:
            return False
        return True
    except Exception as e:
        print(f"缓存校验失败，将重建缓存：{e}")
        return False


def build_preprocessed_cache(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将全部图像和mask预处理成 uint8 memmap。
    优点：
    - 避免每个epoch重复 Image.open / resize / palette mask解析；
    - 多模型连续训练时收益很大；
    - 第一次运行会花几分钟，之后自动复用。
    """
    if not Config.use_preprocessed_cache:
        return records

    paths = get_cache_paths()
    cache_dir = paths["dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    h, w = Config.target_size[1], Config.target_size[0]
    n = len(records)

    if cache_is_valid(records):
        print(f"\n✅ 检测到有效预处理缓存，直接复用：{cache_dir}")
        cached_records = []
        for i, r in enumerate(records):
            rr = dict(r)
            rr["cache_index"] = i
            cached_records.append(rr)
        return cached_records

    print("\n" + "=" * 70)
    print("开始构建高速预处理缓存")
    print(f"缓存目录: {cache_dir}")
    print(f"样本数: {n}, shape=({h}, {w}), dtype=uint8")
    print("第一次运行会慢一些；后续训练将直接读取缓存。")
    print("=" * 70)

    # 删除旧缓存，避免尺寸/样本数变化导致残留
    for p in [paths["images"], paths["masks"]]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    images_mm = np.memmap(paths["images"], dtype=np.uint8, mode="w+", shape=(n, h, w))
    masks_mm = np.memmap(paths["masks"], dtype=np.uint8, mode="w+", shape=(n, h, w))

    cached_records = []
    for i, r in enumerate(tqdm(records, desc="构建图像/mask缓存")):
        img = read_image_gray(r["image_path"]).resize(Config.target_size, Image.BILINEAR)
        mask = read_mask_label(r["mask_path"]).resize(Config.target_size, Image.NEAREST)

        img_np = np.asarray(img, dtype=np.uint8)
        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_np = np.clip(mask_np, 0, Config.num_classes - 1).astype(np.uint8)

        images_mm[i] = img_np
        masks_mm[i] = mask_np

        rr = dict(r)
        rr["cache_index"] = i
        cached_records.append(rr)

    images_mm.flush()
    masks_mm.flush()
    del images_mm, masks_mm

    meta = {
        "num_records": n,
        "target_size": list(Config.target_size),
        "height": h,
        "width": w,
        "dtype": "uint8",
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "record_keys": [make_record_key(r) for r in records],
    }
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ 预处理缓存构建完成。")
    return cached_records


def open_cache_memmaps():
    paths = get_cache_paths()
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    n = int(meta["num_records"])
    h = int(meta["height"])
    w = int(meta["width"])
    images = np.memmap(paths["images"], dtype=np.uint8, mode="r", shape=(n, h, w))
    masks = np.memmap(paths["masks"], dtype=np.uint8, mode="r", shape=(n, h, w))
    return images, masks, meta


class MultimodalRemainingOilDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: CharTokenizer,
        target_size: Tuple[int, int] = (512, 512),
        augment: bool = False,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.target_size = target_size
        self.augment = augment

        self.use_cache = bool(Config.use_preprocessed_cache and len(records) > 0 and "cache_index" in records[0])
        self._images_mm = None
        self._masks_mm = None
        self._cache_meta = None

    def __len__(self):
        return len(self.records)

    def __getstate__(self):
        # Windows DataLoader 多进程使用 spawn，memmap 不能直接可靠pickle；worker中懒加载。
        state = self.__dict__.copy()
        state["_images_mm"] = None
        state["_masks_mm"] = None
        state["_cache_meta"] = None
        return state

    def _ensure_cache_open(self):
        if not self.use_cache:
            return
        if self._images_mm is None or self._masks_mm is None:
            self._images_mm, self._masks_mm, self._cache_meta = open_cache_memmaps()

    def _augment_arrays(self, img_np: np.ndarray, mask_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return img_np, mask_np

        if random.random() < 0.5:
            img_np = np.fliplr(img_np)
            mask_np = np.fliplr(mask_np)

        if random.random() < 0.5:
            img_np = np.flipud(img_np)
            mask_np = np.flipud(mask_np)

        k = random.randint(0, 3)
        if k > 0:
            img_np = np.rot90(img_np, k)
            mask_np = np.rot90(mask_np, k)

        # flip/rot90 可能产生负 stride，torch.from_numpy 不支持，所以 copy
        return np.ascontiguousarray(img_np), np.ascontiguousarray(mask_np)

    def _load_from_cache(self, r: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_cache_open()
        ci = int(r["cache_index"])
        img_np = np.array(self._images_mm[ci], copy=False)
        mask_np = np.array(self._masks_mm[ci], copy=False)
        return img_np, mask_np

    def _load_from_files(self, r: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        img = read_image_gray(r["image_path"]).resize(self.target_size, Image.BILINEAR)
        mask = read_mask_label(r["mask_path"]).resize(self.target_size, Image.NEAREST)
        img_np = np.asarray(img, dtype=np.uint8)
        mask_np = np.asarray(mask, dtype=np.uint8)
        return img_np, mask_np

    def __getitem__(self, idx):
        r = self.records[idx]

        if self.use_cache:
            img_np, mask_np = self._load_from_cache(r)
        else:
            img_np, mask_np = self._load_from_files(r)

        img_np, mask_np = self._augment_arrays(img_np, mask_np)

        # uint8 -> float32 [0,1]，仍在CPU侧完成；后续可通过 channels_last 加速GPU卷积。
        img_float = img_np.astype(np.float32) / 255.0
        img_3ch = np.stack([img_float, img_float, img_float], axis=0)  # [3,H,W]

        mask_np = np.clip(mask_np.astype(np.int64), 0, Config.num_classes - 1)

        input_ids, attention_mask = self.tokenizer.encode(r["text"])

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img_3ch)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_np)).long(),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sample": r["sample"],
            "slice_id": r["slice_id"],
            "image_name": r["image_filename"],
            "text": r["text"],
        }


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    if Config.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = Config.prefetch_factor

    # Windows 下部分环境对 pin_memory_device 支持不稳定，不强制设置。
    return DataLoader(**kwargs)


# ============================================================
# 4. 模型模块
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6)):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.GroupNorm(8, reduction_dim),
                nn.ReLU(inplace=True),
            )
            for bin_size in bins
        ])
        total_dim = in_dim + len(bins) * reduction_dim
        self.out = nn.Sequential(
            nn.Conv2d(total_dim, in_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, in_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        outs = [x]
        for f in self.features:
            y = f(x)
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            outs.append(y)
        return self.out(torch.cat(outs, dim=1))


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x, key_padding_mask=None):
        x1 = self.norm1(x)
        attn_out, _ = self.attn(x1, x1, x1, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class ViTEncoder(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=192, depth=4, num_heads=6, mlp_ratio=4.0, drop_rate=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        grid = img_size // patch_size
        num_patches = grid * grid
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # CNN skip 分支
        self.stem1 = DoubleConv(in_chans, 32)      # 512
        self.pool1 = nn.MaxPool2d(2)
        self.stem2 = DoubleConv(32, 64)            # 256
        self.pool2 = nn.MaxPool2d(2)
        self.stem3 = DoubleConv(64, 128)           # 128
        self.pool3 = nn.MaxPool2d(2)
        self.stem4 = DoubleConv(128, 256)          # 64

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        s1 = self.stem1(x)
        s2 = self.stem2(self.pool1(s1))
        s3 = self.stem3(self.pool2(s2))
        s4 = self.stem4(self.pool3(s3))

        v = self.patch_embed(x)
        B, C, H, W = v.shape
        v = v.flatten(2).transpose(1, 2)
        v = v + self.pos_embed
        v = self.pos_drop(v)

        features = []
        take_indices = {0, max(0, len(self.blocks) // 2), len(self.blocks) - 1}

        for i, blk in enumerate(self.blocks):
            v = blk(v)
            if i in take_indices:
                features.append(v)

        while len(features) < 3:
            features.append(v)

        v = self.norm(v)
        feat = v.transpose(1, 2).reshape(B, C, H, W)

        vit_feats = []
        for f in features[:3]:
            vit_feats.append(f.transpose(1, 2).reshape(B, C, H, W))

        return feat, vit_feats, (s1, s2, s3, s4)


class TextTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=192, embed_dim=192, depth=2, num_heads=6, drop_rate=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.drop = nn.Dropout(drop_rate)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids, attention_mask):
        x = self.token_embed(input_ids) + self.pos_embed[:, :input_ids.size(1), :]
        x = self.drop(x)
        key_padding_mask = attention_mask == 0
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


# ============================================================
# 5. 论文主模型与消融模型：MPP-ViT
# ============================================================

class MultiModalViTPPMTransformerUNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 8,
        img_size: int = 512,
        patch_size: int = 16,
        use_text: bool = True,
        use_ppm: bool = True,
        use_ppm_before_fusion: bool = False,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_ppm = use_ppm
        self.use_ppm_before_fusion = use_ppm_before_fusion

        self.visual_encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=Config.in_channels,
            embed_dim=Config.embed_dim,
            depth=Config.depth,
            num_heads=Config.num_heads,
            mlp_ratio=Config.mlp_ratio,
            drop_rate=Config.drop_rate,
        )

        if use_text:
            self.text_encoder = TextTransformerEncoder(
                vocab_size=vocab_size,
                max_len=Config.max_text_len,
                embed_dim=Config.text_embed_dim,
                depth=Config.text_depth,
                num_heads=Config.text_num_heads,
                drop_rate=Config.text_drop_rate,
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=Config.embed_dim,
                num_heads=Config.num_heads,
                dropout=Config.drop_rate,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(Config.embed_dim)
            self.text_alpha = nn.Parameter(torch.tensor(Config.text_alpha_init))
        else:
            self.text_encoder = None
            self.cross_attn = None
            self.cross_norm = None
            self.text_alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        if use_ppm:
            # 普通PPM：不使用TG-SPPM，不使用文本引导尺度选择
            self.ppm = PPM(in_dim=Config.embed_dim, reduction_dim=48, bins=(1, 2, 3, 6))
            self.ppm_gate = None
        else:
            self.ppm = None
            self.ppm_gate = None

        self.vit_to_s4 = nn.Conv2d(Config.embed_dim, 256, kernel_size=1)
        self.vit_to_s3 = nn.Conv2d(Config.embed_dim, 128, kernel_size=1)
        self.vit_to_s2 = nn.Conv2d(Config.embed_dim, 64, kernel_size=1)

        self.alpha_s4 = nn.Parameter(torch.tensor(0.3))
        self.alpha_s3 = nn.Parameter(torch.tensor(0.3))
        self.alpha_s2 = nn.Parameter(torch.tensor(0.3))

        self.bridge = DoubleConv(Config.embed_dim, 256)
        self.dec4 = UpBlock(256, 256, 256)
        self.dec3 = UpBlock(256, 128, 128)
        self.dec2 = UpBlock(128, 64, 64)
        self.dec1 = UpBlock(64, 32, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, image, input_ids=None, attention_mask=None):
        feat, vit_feats, skips = self.visual_encoder(image)
        s1, s2, s3, s4 = skips
        v1, v2, v3 = vit_feats

        text_feats = None
        key_padding_mask = None

        # 1) 先编码文本，但暂不融合；真正的 Cross-Attention 放在 PPM 之后。
        if self.use_text and input_ids is not None and attention_mask is not None:
            text_feats = self.text_encoder(input_ids, attention_mask)
            key_padding_mask = attention_mask == 0

        # 2) 按论文原始视觉分支逻辑：ViT 后直接接普通 PPM。
        #    注意：这里是最开始的PPM，不是TG-SPPM，不做文本引导尺度选择。
        if self.use_ppm:
            feat = feat + Config.ppm_scale * self.ppm(feat)

        # 3) PPM视觉多尺度增强之后，再和文本做 Cross-Attention 图文融合。
        if self.use_text and text_feats is not None:
            B, C, H, W = feat.shape
            visual_tokens = feat.flatten(2).transpose(1, 2)
            fused_tokens, _ = self.cross_attn(
                query=self.cross_norm(visual_tokens),
                key=text_feats,
                value=text_feats,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            visual_tokens = visual_tokens + self.text_alpha * fused_tokens
            feat = visual_tokens.transpose(1, 2).reshape(B, C, H, W)

        # ViT中间层补充到skip
        v_s2 = F.interpolate(self.vit_to_s2(v1), size=s2.shape[-2:], mode="bilinear", align_corners=False)
        v_s3 = F.interpolate(self.vit_to_s3(v2), size=s3.shape[-2:], mode="bilinear", align_corners=False)
        v_s4 = F.interpolate(self.vit_to_s4(v3), size=s4.shape[-2:], mode="bilinear", align_corners=False)

        s2 = s2 + self.alpha_s2 * v_s2
        s3 = s3 + self.alpha_s3 * v_s3
        s4 = s4 + self.alpha_s4 * v_s4

        x = self.bridge(feat)  # 32x32
        x = self.dec4(x, s4)   # 64x64
        x = self.dec3(x, s3)   # 128x128
        x = self.dec2(x, s2)   # 256x256
        x = self.dec1(x, s1)   # 512x512
        return self.final_conv(x)


# ============================================================
# 6. 对比模型
# ============================================================

class UNetBaseline(nn.Module):
    def __init__(self, num_classes=8, use_ppm=False):
        super().__init__()
        self.use_ppm = use_ppm

        self.inc = DoubleConv(Config.in_channels, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 384)

        if use_ppm:
            self.ppm = PPM(384, 96, bins=(1, 2, 3, 6))
        else:
            self.ppm = None

        self.up4 = UpBlock(384, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, image, input_ids=None, attention_mask=None):
        s1 = self.inc(image)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        x = self.down4(s4)

        if self.use_ppm:
            x = x + Config.ppm_scale * self.ppm(x)

        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.outc(x)


def build_model(model_type: str, vocab_size: int) -> nn.Module:
    if model_type in ["ppm_before_fusion_full", "mpp_vit_full_ppm_before_fusion"]:
        return MultiModalViTPPMTransformerUNet(
            vocab_size=vocab_size,
            num_classes=Config.num_classes,
            img_size=Config.target_size[0],
            patch_size=Config.patch_size,
            use_text=True,
            use_ppm=True,
            use_ppm_before_fusion=True,
        )

    if model_type in ["ppm_before_fusion_no_text", "mpp_vit_no_text_ppm_before_fusion"]:
        return MultiModalViTPPMTransformerUNet(
            vocab_size=vocab_size,
            num_classes=Config.num_classes,
            img_size=Config.target_size[0],
            patch_size=Config.patch_size,
            use_text=False,
            use_ppm=True,
            use_ppm_before_fusion=True,
        )

    if model_type == "mpp_vit_no_ppm":
        return MultiModalViTPPMTransformerUNet(
            vocab_size=vocab_size,
            num_classes=Config.num_classes,
            img_size=Config.target_size[0],
            patch_size=Config.patch_size,
            use_text=True,
            use_ppm=False,
            use_ppm_before_fusion=False,
        )

    if model_type == "mpp_vit_no_text_no_ppm":
        return MultiModalViTPPMTransformerUNet(
            vocab_size=vocab_size,
            num_classes=Config.num_classes,
            img_size=Config.target_size[0],
            patch_size=Config.patch_size,
            use_text=False,
            use_ppm=False,
            use_ppm_before_fusion=False,
        )

    # 兼容旧名称：原full普通PPM也按PPM-before-fusion逻辑运行
    if model_type in ["mpp_vit_full", "mpp_vit_full_plain_ppm"]:
        return MultiModalViTPPMTransformerUNet(
            vocab_size=vocab_size,
            num_classes=Config.num_classes,
            img_size=Config.target_size[0],
            patch_size=Config.patch_size,
            use_text=True,
            use_ppm=True,
            use_ppm_before_fusion=True,
        )

    if model_type == "mpp_vit_no_text":
        return MultiModalViTPPMTransformerUNet(
            vocab_size=vocab_size,
            num_classes=Config.num_classes,
            img_size=Config.target_size[0],
            patch_size=Config.patch_size,
            use_text=False,
            use_ppm=True,
            use_ppm_before_fusion=True,
        )

    if model_type == "transunet":
        # 视觉Transformer + U-Net式解码，但无文本、无PPM
        return MultiModalViTPPMTransformerUNet(
            vocab_size=vocab_size,
            num_classes=Config.num_classes,
            img_size=Config.target_size[0],
            patch_size=Config.patch_size,
            use_text=False,
            use_ppm=False,
        )

    if model_type == "unet":
        return UNetBaseline(num_classes=Config.num_classes, use_ppm=False)

    if model_type == "cnn_ppm_unet":
        return UNetBaseline(num_classes=Config.num_classes, use_ppm=True)

    raise ValueError(f"未知 model_type: {model_type}")


# ============================================================
# 7. 损失函数与指标
# ============================================================

class WeightedDiceLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        self.register_buffer("class_weights", class_weights.float())

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims)

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        weighted = dice * self.class_weights
        loss = 1.0 - weighted.sum() / (self.class_weights.sum() + 1e-8)
        return loss


class SegMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.correct = 0
        self.total = 0

    def _fast_hist(self, true, pred):
        mask = (true >= 0) & (true < self.num_classes)
        hist = np.bincount(
            self.num_classes * true[mask].astype(int) + pred[mask].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        self.correct += int((pred == target).sum().item())
        self.total += int(pred.numel())

        for p, t in zip(pred_np, target_np):
            self.hist += self._fast_hist(t.flatten(), p.flatten())

    def compute(self) -> Dict[str, Any]:
        hist = self.hist
        diag = np.diag(hist).astype(np.float64)

        gt = hist.sum(axis=1).astype(np.float64)
        pred = hist.sum(axis=0).astype(np.float64)
        union = gt + pred - diag

        iou = diag / (union + 1e-8)
        dice = 2 * diag / (gt + pred + 1e-8)
        precision = diag / (pred + 1e-8)
        recall = diag / (gt + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "accuracy": self.correct / self.total if self.total > 0 else 0.0,
            "mIoU": float(np.nanmean(iou)),
            "mDice": float(np.nanmean(dice)),
            "mPrecision": float(np.nanmean(precision)),
            "mRecall": float(np.nanmean(recall)),
            "mF1": float(np.nanmean(f1)),
            "per_class": {
                int(k): {
                    "class_name": Config.class_names[k],
                    "iou": float(iou[k]),
                    "dice": float(dice[k]),
                    "precision": float(precision[k]),
                    "recall": float(recall[k]),
                    "f1": float(f1[k]),
                    "gt_pixels": int(gt[k]),
                    "pred_pixels": int(pred[k]),
                }
                for k in range(self.num_classes)
            },
            "confusion_matrix": hist.tolist(),
        }


def compute_auto_class_weights(train_records: List[Dict[str, Any]], max_samples: int = 600) -> torch.Tensor:
    print("\n计算类别权重...")
    rng = random.Random(Config.seed)
    records = train_records.copy()
    rng.shuffle(records)
    records = records[:min(max_samples, len(records))]

    counts = np.zeros(Config.num_classes, dtype=np.float64)

    use_cache = (
        Config.use_preprocessed_cache
        and Config.use_cache_for_class_weights
        and len(records) > 0
        and "cache_index" in records[0]
        and get_cache_paths()["meta"].exists()
    )

    if use_cache:
        print("使用预处理缓存统计类别权重，避免重复读取 TIFF。")
        _, masks_mm, _ = open_cache_memmaps()
        for r in tqdm(records, desc="统计训练集标签频次(cache)"):
            arr = np.asarray(masks_mm[int(r["cache_index"])], dtype=np.int64)
            arr = np.clip(arr, 0, Config.num_classes - 1)
            counts += np.bincount(arr.ravel(), minlength=Config.num_classes)
        del masks_mm
    else:
        for r in tqdm(records, desc="统计训练集标签频次"):
            mask = read_mask_label(r["mask_path"])
            mask = mask.resize(Config.target_size, Image.NEAREST)
            arr = np.array(mask).astype(np.int64)
            arr = np.clip(arr, 0, Config.num_classes - 1)
            counts += np.bincount(arr.ravel(), minlength=Config.num_classes)

    freq = counts / (counts.sum() + 1e-8)
    weights = 1.0 / np.sqrt(freq + 1e-8)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.3, 6.0)

    print("类别像素统计:", counts.astype(int).tolist())
    print("自动类别权重:")
    for i, w in enumerate(weights):
        print(f"  {i} {Config.class_names[i]}: {w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# 8. 训练与测试
# ============================================================

def move_batch_to_device(batch, device):
    if Config.use_channels_last and device.type == "cuda":
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
    else:
        images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    return images, masks, input_ids, attention_mask


def maybe_dropout_text(input_ids, attention_mask):
    if Config.text_dropout_prob <= 0:
        return input_ids, attention_mask
    if np.random.rand() < Config.text_dropout_prob:
        input_ids = input_ids.clone()
        attention_mask = attention_mask.clone()
        input_ids[:, 1:] = 0
        attention_mask[:, 1:] = 0
    return input_ids, attention_mask


def forward_loss(model, batch, device, ce_loss_fn, dice_loss_fn, amp_enabled=True):
    images, masks, input_ids, attention_mask = move_batch_to_device(batch, device)

    with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype(), enabled=(amp_enabled and device.type == "cuda")):
        outputs = model(images, input_ids, attention_mask)
        loss_ce = ce_loss_fn(outputs, masks)
        loss_dice = dice_loss_fn(outputs, masks)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

    return loss, loss_ce.detach(), loss_dice.detach(), outputs, masks


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, ce_loss_fn, dice_loss_fn, epoch, exp_name):
    model.train()
    metrics = SegMetrics(Config.num_classes)
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"{exp_name} | Train {epoch}")
    for step, batch in enumerate(pbar, start=1):
        batch["input_ids"], batch["attention_mask"] = maybe_dropout_text(batch["input_ids"], batch["attention_mask"])

        loss, loss_ce, loss_dice, outputs, masks = forward_loss(
            model, batch, device, ce_loss_fn, dice_loss_fn, Config.use_amp
        )
        loss_to_backward = loss / Config.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if step % Config.grad_accum_steps == 0 or step == len(loader):
            if Config.clip_grad_norm is not None and Config.clip_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

        total_loss += float(loss.item())
        total_ce += float(loss_ce.item())
        total_dice += float(loss_dice.item())

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ce": f"{loss_ce.item():.4f}",
            "dice": f"{loss_dice.item():.4f}",
        })

    if scheduler is not None:
        scheduler.step()

    results = metrics.compute()
    n = max(1, len(loader))
    results.update({
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "dice_loss": total_dice / n,
    })
    return results


@torch.no_grad()
def evaluate_model(model, loader, device, ce_loss_fn, dice_loss_fn, desc="Eval"):
    model.eval()
    metrics = SegMetrics(Config.num_classes)
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc=desc)
    for batch in pbar:
        loss, loss_ce, loss_dice, outputs, masks = forward_loss(
            model, batch, device, ce_loss_fn, dice_loss_fn, Config.use_amp
        )
        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, masks)

        total_loss += float(loss.item())
        total_ce += float(loss_ce.item())
        total_dice += float(loss_dice.item())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    results = metrics.compute()
    n = max(1, len(loader))
    results.update({
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "dice_loss": total_dice / n,
    })
    return results


def save_metrics(metrics: Dict[str, Any], exp_dir: Path, split_name: str):
    exp_dir.mkdir(parents=True, exist_ok=True)

    json_path = exp_dir / f"{split_name}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_path = exp_dir / f"{split_name}_per_class_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "iou", "dice", "precision", "recall", "f1", "gt_pixels", "pred_pixels"])
        for k, v in metrics["per_class"].items():
            writer.writerow([
                k, v["class_name"], v["iou"], v["dice"], v["precision"],
                v["recall"], v["f1"], v["gt_pixels"], v["pred_pixels"]
            ])


def plot_history(history: Dict[str, List[float]], exp_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_mIoU"], label="train mIoU")
    axes[1].plot(history["val_mIoU"], label="val mIoU")
    axes[1].set_title("mIoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["train_mDice"], label="train mDice")
    axes[2].plot(history["val_mDice"], label="val mDice")
    axes[2].set_title("mDice")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(exp_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_history_csv(history: Dict[str, List[float]], exp_dir: Path):
    df = pd.DataFrame(history)
    df.to_csv(exp_dir / "history.csv", index=False, encoding="utf-8-sig")


class TrainingStopped(Exception):
    """用于安全暂停训练流程。"""


def get_stop_file_path() -> Path:
    return Config.output_dir / Config.stop_file_name


def cleanup_stale_stop_file():
    """
    上次安全暂停后 STOP 文件会留在目录里。
    下次启动时自动删除，避免一恢复训练就再次暂停。
    """
    stop_file = get_stop_file_path()
    if stop_file.exists():
        try:
            stop_file.unlink()
            print(f"已删除上次遗留的暂停文件：{stop_file}")
        except Exception as e:
            print(f"警告：无法删除暂停文件 {stop_file}，原因：{e}")


def stop_requested() -> bool:
    return get_stop_file_path().exists()


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch,
    best_val_miou,
    tokenizer,
    exp_cfg,
    scheduler=None,
    scaler=None,
    history=None,
    best_epoch=-1,
    patience_counter=0,
    extra: Optional[Dict[str, Any]] = None,
):
    """
    通用checkpoint保存函数。
    best_model.pth / final_model.pth / resume_checkpoint.pth 都可使用。
    resume_checkpoint.pth 会额外保存 scheduler、scaler、history、patience 等状态。
    """
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_miou": float(best_val_miou),
        "best_epoch": int(best_epoch) if best_epoch is not None else -1,
        "patience_counter": int(patience_counter),
        "history": history,
        "vocab_size": tokenizer.vocab_size,
        "max_text_len": tokenizer.max_len,
        "class_names": Config.class_names,
        "experiment": exp_cfg,
        "saved_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_resume_checkpoint(
    resume_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
) -> Optional[Dict[str, Any]]:
    if not resume_path.exists():
        return None

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt


def safe_write_all_summaries(summaries: List[Dict[str, Any]]):
    if not summaries:
        return
    Config.result_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summaries)
    summary_path = Config.result_dir / "all_experiments_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"已更新实验汇总表：{summary_path}")


def train_experiment(
    exp_cfg: Dict[str, str],
    tokenizer: CharTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    exp_name = exp_cfg["name"]
    model_type = exp_cfg["model_type"]

    print("\n" + "=" * 90)
    print(f"开始实验：{exp_name} | model_type={model_type}")
    print("=" * 90)

    exp_model_dir = Config.model_dir / exp_name
    exp_result_dir = Config.result_dir / exp_name
    exp_pred_dir = Config.pred_dir / exp_name
    for d in [exp_model_dir, exp_result_dir, exp_pred_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary_json = exp_result_dir / "summary.json"
    best_model_path = exp_model_dir / "best_model.pth"
    resume_path = exp_model_dir / Config.resume_checkpoint_name

    # 如果该实验已经完整完成，默认跳过，避免重复训练。
    if Config.skip_completed_experiments and summary_json.exists() and best_model_path.exists():
        print(f"检测到 {exp_name} 已完成，跳过训练并读取已有 summary.json。")
        with open(summary_json, "r", encoding="utf-8") as f:
            return json.load(f)

    tokenizer.save(exp_model_dir / "tokenizer.json")

    model = build_model(model_type=model_type, vocab_size=tokenizer.vocab_size).to(device)
    if Config.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if Config.use_torch_compile and device.type == "cuda":
        try:
            print(f"启用 torch.compile(mode={Config.torch_compile_mode})，第一次 epoch 可能较慢。")
            model = torch.compile(model, mode=Config.torch_compile_mode)
        except Exception as e:
            print(f"torch.compile 启用失败，自动回退普通模式：{e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: total={total_params:,}, trainable={trainable_params:,}")

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    dice_loss_fn = WeightedDiceLoss(Config.num_classes, class_weights.to(device)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)

    amp_dtype = autocast_dtype()
    scaler = None
    if Config.use_amp and device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler(device="cuda")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_mIoU": [],
        "val_mIoU": [],
        "train_mDice": [],
        "val_mDice": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_miou = -1.0
    best_epoch = -1
    patience_counter = 0
    start_epoch = 1

    # --------------------------------------------------------
    # 自动断点续训
    # --------------------------------------------------------
    if Config.auto_resume and resume_path.exists():
        ckpt = load_resume_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)
        if ckpt is not None:
            loaded_epoch = int(ckpt.get("epoch", 0))
            start_epoch = loaded_epoch + 1
            best_val_miou = float(ckpt.get("best_val_miou", -1.0))
            best_epoch = int(ckpt.get("best_epoch", -1))
            patience_counter = int(ckpt.get("patience_counter", 0))
            if ckpt.get("history") is not None:
                history = ckpt["history"]

            print(
                f"🔁 检测到断点，自动从 {exp_name} 的 epoch {start_epoch} 继续。"
                f" 当前 best_val_mIoU={best_val_miou:.4f}, best_epoch={best_epoch}, "
                f"早停计数={patience_counter}/{Config.patience}"
            )

    if start_epoch > Config.num_epochs:
        print(f"{exp_name} 已经训练到 epoch {start_epoch-1}，超过 num_epochs={Config.num_epochs}。直接进入测试。")

    try:
        for epoch in range(start_epoch, Config.num_epochs + 1):
            t0 = time.time()

            train_res = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler, device,
                ce_loss_fn, dice_loss_fn, epoch, exp_name
            )
            val_res = evaluate_model(
                model, val_loader, device, ce_loss_fn, dice_loss_fn,
                desc=f"{exp_name} | Val {epoch}"
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_res["loss"])
            history["val_loss"].append(val_res["loss"])
            history["train_mIoU"].append(train_res["mIoU"])
            history["val_mIoU"].append(val_res["mIoU"])
            history["train_mDice"].append(train_res["mDice"])
            history["val_mDice"].append(val_res["mDice"])
            history["train_acc"].append(train_res["accuracy"])
            history["val_acc"].append(val_res["accuracy"])

            print(
                f"[{exp_name}] Epoch {epoch}/{Config.num_epochs} "
                f"time={time.time()-t0:.1f}s | "
                f"train_loss={train_res['loss']:.4f}, train_mIoU={train_res['mIoU']:.4f}, train_mDice={train_res['mDice']:.4f} | "
                f"val_loss={val_res['loss']:.4f}, val_mIoU={val_res['mIoU']:.4f}, val_mDice={val_res['mDice']:.4f}"
            )

            save_history_csv(history, exp_result_dir)
            plot_history(history, exp_result_dir)

            if val_res["mIoU"] > best_val_miou:
                best_val_miou = val_res["mIoU"]
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    exp_model_dir / "best_model.pth",
                    model, optimizer, epoch, best_val_miou, tokenizer, exp_cfg,
                    scheduler=scheduler, scaler=scaler, history=history,
                    best_epoch=best_epoch, patience_counter=patience_counter,
                    extra={"checkpoint_type": "best"}
                )
                print(f"✅ 保存最佳模型：epoch={epoch}, val_mIoU={best_val_miou:.4f}")
            else:
                patience_counter += 1
                print(f"早停计数：{patience_counter}/{Config.patience}")

            if Config.save_resume_every_epoch:
                save_checkpoint(
                    resume_path,
                    model, optimizer, epoch, best_val_miou, tokenizer, exp_cfg,
                    scheduler=scheduler, scaler=scaler, history=history,
                    best_epoch=best_epoch, patience_counter=patience_counter,
                    extra={"checkpoint_type": "resume"}
                )
                print(f"💾 已保存断点：{resume_path}")

            # 安全暂停：用户创建 STOP_TRAINING.txt 后，在当前 epoch 完成后暂停。
            if stop_requested():
                save_checkpoint(
                    resume_path,
                    model, optimizer, epoch, best_val_miou, tokenizer, exp_cfg,
                    scheduler=scheduler, scaler=scaler, history=history,
                    best_epoch=best_epoch, patience_counter=patience_counter,
                    extra={"checkpoint_type": "resume", "stop_reason": "STOP_TRAINING.txt"}
                )
                print("\n🛑 检测到 STOP_TRAINING.txt，已在当前 epoch 结束后安全保存断点。")
                print(f"下次直接重新运行脚本，将从 {exp_name} 的 epoch {epoch+1} 继续。")
                raise TrainingStopped(f"用户请求暂停：{exp_name} epoch {epoch}")

            if patience_counter >= Config.patience:
                print(f"⏹️ 早停触发：epoch={epoch}")
                break

    except KeyboardInterrupt:
        # Ctrl+C 只能保证保存当前内存里的模型状态；最稳的方式仍然是 STOP_TRAINING.txt。
        current_epoch = history["epoch"][-1] if history["epoch"] else max(0, start_epoch - 1)
        save_checkpoint(
            resume_path,
            model, optimizer, current_epoch, best_val_miou, tokenizer, exp_cfg,
            scheduler=scheduler, scaler=scaler, history=history,
            best_epoch=best_epoch, patience_counter=patience_counter,
            extra={"checkpoint_type": "resume", "stop_reason": "KeyboardInterrupt"}
        )
        print("\n🛑 收到 Ctrl+C，已尽量保存断点。")
        print("下次直接重新运行脚本，会从最近保存的 epoch 后继续。")
        raise TrainingStopped(f"KeyboardInterrupt：{exp_name}")

    # 至少有一个epoch完成，或从已有断点直接进入测试。
    final_epoch = history["epoch"][-1] if history["epoch"] else 0
    save_checkpoint(
        exp_model_dir / "final_model.pth",
        model, optimizer, final_epoch, best_val_miou, tokenizer, exp_cfg,
        scheduler=scheduler, scaler=scaler, history=history,
        best_epoch=best_epoch, patience_counter=patience_counter,
        extra={"checkpoint_type": "final"}
    )

    # --------------------------------------------------------
    # 加载最佳模型并在测试集评估
    # --------------------------------------------------------
    if not best_model_path.exists():
        # 极端情况下没有best_model，退回final模型
        print("警告：未找到 best_model.pth，将使用 final_model.pth 进行测试。")
        best_model_path = exp_model_dir / "final_model.pth"

    print(f"\n加载 {exp_name} 最佳模型进行测试评估...")
    best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_res = evaluate_model(
        model, test_loader, device, ce_loss_fn, dice_loss_fn,
        desc=f"{exp_name} | Test"
    )
    save_metrics(test_res, exp_result_dir, "test")

    # 保存少量测试可视化
    visualize_predictions(model, test_loader, device, exp_pred_dir, num_batches=2)

    summary = {
        "experiment": exp_name,
        "model_type": model_type,
        "fusion_order": "ViT -> Plain PPM -> Cross-Attention -> Decoder",
        "best_epoch": best_epoch,
        "best_val_mIoU": best_val_miou,
        "test_accuracy": test_res["accuracy"],
        "test_mIoU": test_res["mIoU"],
        "test_mDice": test_res["mDice"],
        "test_mPrecision": test_res["mPrecision"],
        "test_mRecall": test_res["mRecall"],
        "test_mF1": test_res["mF1"],
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n测试结果：")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


@torch.no_grad()
def visualize_predictions(model, loader, device, out_dir: Path, num_batches: int = 2):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    # 自定义调色板，便于看图
    cmap = np.array([
        [0, 0, 0],
        [150, 150, 150],
        [0, 102, 255],
        [255, 230, 0],
        [255, 140, 0],
        [140, 216, 255],
        [255, 0, 0],
        [160, 32, 240],
    ], dtype=np.uint8)

    saved = 0
    for bidx, batch in enumerate(loader):
        if bidx >= num_batches:
            break

        images, masks, input_ids, attention_mask = move_batch_to_device(batch, device)
        outputs = model(images, input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()
        images_np = images.detach().cpu().numpy()

        bs = images_np.shape[0]
        for i in range(bs):
            img_gray = images_np[i, 0]
            gt_rgb = cmap[np.clip(masks_np[i], 0, Config.num_classes - 1)]
            pred_rgb = cmap[np.clip(preds[i], 0, Config.num_classes - 1)]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_gray, cmap="gray")
            axes[0].set_title("原图")
            axes[0].axis("off")

            axes[1].imshow(gt_rgb)
            axes[1].set_title("真实标签")
            axes[1].axis("off")

            axes[2].imshow(pred_rgb)
            axes[2].set_title("预测")
            axes[2].axis("off")

            title = f"{batch['sample'][i]}_filtered{int(batch['slice_id'][i]):03d}"
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(out_dir / f"{title}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved += 1

    print(f"已保存测试可视化 {saved} 张到 {out_dir}")



# ============================================================
# 9.1 多随机种子统计性检验工具
# ============================================================

def _load_summary_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def maybe_load_existing_seed42_summary(model_group: str) -> Optional[Dict[str, Any]]:
    """
    复用已经完成的主实验结果，避免把seed=42的MPP-ViT/TransUNet再训练一遍。
    默认查找原始主实验输出目录：
      multimodal_remaining_oil_outputs_fast_ppm_before_fusion/results/01_PPM_before_Fusion_full/summary.json
      multimodal_remaining_oil_outputs_fast_ppm_before_fusion/results/07_TransUNet_image_only/summary.json
    如果你想强制重新训练seed=42，把Config.reuse_existing_seed42_results改为False。
    """
    if not getattr(Config, "reuse_existing_seed42_results", True):
        return None

    old_root = Path("multimodal_remaining_oil_outputs_fast_ppm_before_fusion") / "results"
    old_exp = {
        "MPP-ViT": "01_PPM_before_Fusion_full",
        "TransUNet": "07_TransUNet_image_only",
    }.get(model_group)
    if old_exp is None:
        return None

    s = _load_summary_json(old_root / old_exp / "summary.json")
    if s is None:
        return None

    s = dict(s)
    s["source"] = "reused_existing_seed42"
    return s


def add_repeat_metadata(summary: Dict[str, Any], seed: int, model_group: str, exp_name: str) -> Dict[str, Any]:
    summary = dict(summary)
    summary["seed"] = int(seed)
    summary["model_group"] = model_group
    summary["experiment"] = exp_name
    return summary


def write_repeat_summary(summaries: List[Dict[str, Any]]):
    Config.result_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summaries)
    if len(df) > 0:
        df = df.sort_values(["seed", "model_group"])
    out = Config.result_dir / "repeat_seed_all_summaries.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"多随机种子结果汇总已保存：{out}")
    return df


def exact_sign_flip_pvalue(diff: np.ndarray) -> float:
    """
    小样本配对差值的双侧sign-flip exact p-value。
    n=3时比只报t-test更稳妥；该检验不假设差值正态。
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n == 0:
        return float("nan")
    obs = abs(float(np.mean(diff)))
    count = 0
    total = 2 ** n
    for mask in range(total):
        signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=float)
        val = abs(float(np.mean(signs * diff)))
        if val >= obs - 1e-15:
            count += 1
    return count / total


def run_paired_statistical_tests(summaries: List[Dict[str, Any]]):
    """
    对MPP-ViT和TransUNet的多随机种子结果做配对统计检验。
    输出：
      - repeat_seed_paired_values.csv：每个seed的成对指标
      - repeat_seed_statistical_tests.csv：均值±标准差、差值、置信区间、p值、效应量
    """
    df = pd.DataFrame(summaries)
    if df.empty:
        print("没有可用于统计检验的summary。")
        return

    metrics = ["test_mIoU", "test_mDice", "test_accuracy", "test_mPrecision", "test_mRecall", "test_mF1"]
    paired_rows = []
    for seed in sorted(df["seed"].dropna().unique()):
        sub = df[df["seed"] == seed]
        mpp = sub[sub["model_group"] == "MPP-ViT"]
        tru = sub[sub["model_group"] == "TransUNet"]
        if len(mpp) == 0 or len(tru) == 0:
            continue
        row = {"seed": int(seed)}
        for metric in metrics:
            row[f"MPP-ViT_{metric}"] = float(mpp.iloc[0][metric])
            row[f"TransUNet_{metric}"] = float(tru.iloc[0][metric])
            row[f"diff_{metric}"] = row[f"MPP-ViT_{metric}"] - row[f"TransUNet_{metric}"]
        paired_rows.append(row)

    paired_df = pd.DataFrame(paired_rows)
    paired_path = Config.result_dir / "repeat_seed_paired_values.csv"
    paired_df.to_csv(paired_path, index=False, encoding="utf-8-sig")
    print(f"配对seed指标已保存：{paired_path}")

    if len(paired_df) < 3:
        print("警告：有效成对seed少于3个。建议至少完成3个seed后再写入论文。")

    # scipy可用则补充paired t-test和Wilcoxon；不可用则仍输出exact sign-flip p-value。
    try:
        from scipy import stats as scipy_stats
    except Exception:
        scipy_stats = None

    stat_rows = []
    for metric in metrics:
        a = paired_df[f"MPP-ViT_{metric}"].to_numpy(dtype=float)
        b = paired_df[f"TransUNet_{metric}"].to_numpy(dtype=float)
        d = a - b
        d = d[np.isfinite(d)]
        n = len(d)
        if n == 0:
            continue
        mean_a = float(np.mean(a))
        std_a = float(np.std(a, ddof=1)) if n > 1 else 0.0
        mean_b = float(np.mean(b))
        std_b = float(np.std(b, ddof=1)) if n > 1 else 0.0
        mean_diff = float(np.mean(d))
        std_diff = float(np.std(d, ddof=1)) if n > 1 else 0.0
        se = std_diff / math.sqrt(n) if n > 1 else float("nan")
        if n > 1:
            if scipy_stats is not None:
                tcrit = float(scipy_stats.t.ppf(0.975, df=n-1))
            else:
                # n=3时df=2的双侧95% t临界值；其他n无scipy时保守使用1.96
                tcrit = 4.302652729 if n == 3 else 1.96
            ci_low = mean_diff - tcrit * se
            ci_high = mean_diff + tcrit * se
        else:
            ci_low = ci_high = float("nan")

        cohen_dz = mean_diff / std_diff if std_diff > 0 else float("inf")
        exact_p = exact_sign_flip_pvalue(d)

        t_stat = t_p = wilcoxon_stat = wilcoxon_p = float("nan")
        if scipy_stats is not None and n >= 2:
            try:
                tres = scipy_stats.ttest_rel(a, b)
                t_stat = float(tres.statistic)
                t_p = float(tres.pvalue)
            except Exception:
                pass
            try:
                # zero_method="wilcox"遇到全0差值会报错，所以捕获
                wres = scipy_stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
                wilcoxon_stat = float(wres.statistic)
                wilcoxon_p = float(wres.pvalue)
            except Exception:
                pass

        stat_rows.append({
            "metric": metric,
            "n_pairs": n,
            "MPP-ViT_mean": mean_a,
            "MPP-ViT_std": std_a,
            "TransUNet_mean": mean_b,
            "TransUNet_std": std_b,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "cohen_dz": cohen_dz,
            "paired_t_stat": t_stat,
            "paired_t_p": t_p,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
            "exact_sign_flip_p": exact_p,
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_path = Config.result_dir / "repeat_seed_statistical_tests.csv"
    stat_df.to_csv(stat_path, index=False, encoding="utf-8-sig")
    with open(Config.result_dir / "repeat_seed_statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stat_rows, f, ensure_ascii=False, indent=2)

    print(f"统计性检验结果已保存：{stat_path}")
    print(stat_df)


# ============================================================
# 9. 主流程
# ============================================================

def save_split_files(train_records, val_records, test_records):
    split_dir = Config.result_dir / "dataset_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    def write_records(path: Path, rows: List[Dict[str, Any]]):
        fields = ["sample", "slice_id", "image_filename", "mask_filename", "image_path", "mask_path", "text"]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fields})

    write_records(split_dir / "train_records.csv", train_records)
    write_records(split_dir / "val_records.csv", val_records)
    write_records(split_dir / "test_records.csv", test_records)

    print(f"数据划分文件已保存到：{split_dir}")


def main():
    ensure_dirs()

    print("\n多随机种子统计实验说明：")
    print("  本脚本只比较 MPP-ViT 与 TransUNet。")
    print("  默认seed列表为 Config.repeat_seeds。")
    print("  若检测到原始主实验seed=42结果，会直接复用，避免重复训练。")
    print("  其他seed会正常训练，并在全部完成后输出paired t-test、Wilcoxon和exact sign-flip检验。\n")

    cleanup_stale_stop_file()
    device = get_device()

    # 数据只扫描/缓存一次；不同seed主要影响训练集shuffle、数据增强、初始化和优化过程。
    records = collect_records()
    if Config.use_preprocessed_cache:
        records = build_preprocessed_cache(records)

    all_summaries = []

    for seed in Config.repeat_seeds:
        Config.seed = int(seed)
        set_seed(Config.seed)

        print("\n" + "#" * 100)
        print(f"开始随机种子 seed={Config.seed}")
        print("#" * 100)

        train_records, val_records, test_records = split_records_by_sample(records)
        print(f"seed={seed} 划分：train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

        tokenizer = CharTokenizer([r["text"] for r in train_records], max_len=Config.max_text_len)
        print(f"seed={seed} tokenizer vocab_size={tokenizer.vocab_size}")

        train_dataset = MultimodalRemainingOilDataset(train_records, tokenizer, Config.target_size, augment=True)
        val_dataset = MultimodalRemainingOilDataset(val_records, tokenizer, Config.target_size, augment=False)
        test_dataset = MultimodalRemainingOilDataset(test_records, tokenizer, Config.target_size, augment=False)

        train_loader = build_loader(train_dataset, Config.batch_size, shuffle=True, drop_last=True)
        val_loader = build_loader(val_dataset, Config.batch_size, shuffle=False, drop_last=False)
        test_loader = build_loader(test_dataset, Config.batch_size, shuffle=False, drop_last=False)

        if Config.class_weight_strategy == "auto":
            class_weights = compute_auto_class_weights(train_records, Config.class_weight_max_samples)
        else:
            class_weights = torch.tensor(Config.manual_class_weights, dtype=torch.float32)

        seed_experiments = [
            {"name": f"seed_{seed}_01_MPP_ViT_full", "model_type": "ppm_before_fusion_full", "model_group": "MPP-ViT"},
            {"name": f"seed_{seed}_02_TransUNet_image_only", "model_type": "transunet", "model_group": "TransUNet"},
        ]

        for exp_cfg in seed_experiments:
            exp_name = exp_cfg["name"]
            model_group = exp_cfg["model_group"]

            # seed=42优先复用你之前已经训练好的结果，不重复训练。
            reused = None
            if int(seed) == 42:
                reused = maybe_load_existing_seed42_summary(model_group)
            if reused is not None:
                summary = add_repeat_metadata(reused, seed, model_group, exp_name)
                print(f"复用已有seed=42结果：{model_group} -> {summary.get('test_mIoU')}")
            else:
                summary = train_experiment(
                    exp_cfg={"name": exp_name, "model_type": exp_cfg["model_type"]},
                    tokenizer=tokenizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    class_weights=class_weights,
                    device=device,
                )
                summary = add_repeat_metadata(summary, seed, model_group, exp_name)

                # 把seed/model_group写回summary.json，方便之后单独查看。
                summary_json = Config.result_dir / exp_name / "summary.json"
                with open(summary_json, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

            all_summaries = [s for s in all_summaries if not (s.get("seed") == seed and s.get("model_group") == model_group)]
            all_summaries.append(summary)
            write_repeat_summary(all_summaries)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_repeat_summary(all_summaries)
    run_paired_statistical_tests(all_summaries)


if __name__ == "__main__":
    main()
