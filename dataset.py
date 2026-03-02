"""
数据预处理与 PyTorch Dataset。
  - 读取 CHAOS DICOM / Ground-truth PNG
  - 将 T1 InPhase、T1 OutPhase 重采样到 T2 空间
  - 归一化、缓存为 .npz
  - 提供 LiverDataset 供训练 / 验证使用
"""
import os
import glob
import random
import numpy as np
import cv2
import SimpleITK as sitk
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

import config

# ====================== DICOM / PNG 读取 ======================

def _find_subdir(base, keyword):
    """在 base 目录下查找名称包含 keyword 的子目录（大小写不敏感）。"""
    if not os.path.isdir(base):
        return None
    for name in os.listdir(base):
        if keyword.lower() in name.lower() and os.path.isdir(os.path.join(base, name)):
            return os.path.join(base, name)
    return None


def read_dicom_series(dicom_dir):
    """读取单个 DICOM 序列，返回 SimpleITK Image（带空间信息）。"""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"目录中未找到 DICOM 序列: {dicom_dir}")
    fnames = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(fnames)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def read_ground_pngs(ground_dir):
    """读取 Ground-truth PNG，按文件名中的数字排序，返回 (D, H, W) uint8。"""
    files = glob.glob(os.path.join(ground_dir, "*.png"))
    if not files:
        raise FileNotFoundError(f"未找到 PNG: {ground_dir}")

    def _sort_key(f):
        base = os.path.splitext(os.path.basename(f))[0]
        nums = [c for c in base.split("-") if c.isdigit()]
        return int(nums[-1]) if nums else 0

    files = sorted(files, key=_sort_key)
    slices = [np.array(Image.open(f).convert("L"), dtype=np.uint8) for f in files]
    return np.stack(slices, axis=0)


def extract_liver_mask(gt_volume):
    """从多器官标注中提取肝脏二值 mask。"""
    return ((gt_volume >= config.LIVER_LO) &
            (gt_volume <= config.LIVER_HI)).astype(np.float32)


# ====================== 重采样 / 归一化 ======================

def resample_to_reference(source_img, ref_img, is_label=False):
    """将 source SimpleITK Image 重采样到 ref 的空间网格。"""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(source_img)


def normalize(vol):
    """逐 volume z-score 归一化。"""
    mu, sigma = vol.mean(), vol.std()
    return (vol - mu) / max(sigma, 1e-8)


def resize_slice(img, size, is_mask=False):
    """将 2-D 切片 resize 到 (size, size)。"""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


# ====================== 单 case 预处理 ======================

def preprocess_case(case_id):
    """
    读取一个 case 的所有序列，重采样 T1 到 T2 空间，
    归一化、resize，返回 dict 供缓存。
    """
    cdir = os.path.join(config.TRAIN_ROOT, str(case_id))

    # ----- T2 SPIR（参考空间）-----
    t2_img = read_dicom_series(os.path.join(cdir, "T2SPIR", "DICOM_anon"))
    t2_arr = sitk.GetArrayFromImage(t2_img).astype(np.float32)
    t2_gt = read_ground_pngs(os.path.join(cdir, "T2SPIR", "Ground"))
    liver = extract_liver_mask(t2_gt)

    # ----- T1 DUAL -----
    t1_dcm = os.path.join(cdir, "T1DUAL", "DICOM_anon")
    t1in_dir = _find_subdir(t1_dcm, "in")
    t1out_dir = _find_subdir(t1_dcm, "out")
    if t1in_dir is None or t1out_dir is None:
        raise FileNotFoundError(f"T1DUAL InPhase/OutPhase 未找到: {t1_dcm}")

    t1in_img = read_dicom_series(t1in_dir)
    t1out_img = read_dicom_series(t1out_dir)

    # 重采样到 T2 空间
    t1in_res = sitk.GetArrayFromImage(
        resample_to_reference(t1in_img, t2_img)).astype(np.float32)
    t1out_res = sitk.GetArrayFromImage(
        resample_to_reference(t1out_img, t2_img)).astype(np.float32)

    # 归一化
    t2_arr = normalize(t2_arr)
    t1in_res = normalize(t1in_res)
    t1out_res = normalize(t1out_res)

    sz = config.IMAGE_SIZE
    D = t2_arr.shape[0]
    t2_out = np.zeros((D, sz, sz), dtype=np.float32)
    t1in_out = np.zeros_like(t2_out)
    t1out_out = np.zeros_like(t2_out)
    mask_out = np.zeros_like(t2_out)

    for i in range(D):
        t2_out[i] = resize_slice(t2_arr[i], sz)
        t1in_out[i] = resize_slice(t1in_res[i], sz)
        t1out_out[i] = resize_slice(t1out_res[i], sz)
        mask_out[i] = resize_slice(liver[i], sz, is_mask=True)

    return dict(t2=t2_out, t1_inphase=t1in_out,
                t1_outphase=t1out_out, liver_mask=mask_out)


def preprocess_all():
    """预处理全部训练 case 并保存到缓存目录。"""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    from tqdm import tqdm
    for cid in tqdm(config.CASE_IDS, desc="预处理"):
        out_path = os.path.join(config.CACHE_DIR, f"case_{cid}.npz")
        if os.path.exists(out_path):
            print(f"  case {cid} 已缓存，跳过")
            continue
        try:
            data = preprocess_case(cid)
            np.savez_compressed(out_path, **data)
            print(f"  case {cid}: {data['t2'].shape[0]} slices")
        except Exception as e:
            print(f"  [ERROR] case {cid}: {e}")


# ====================== 数据增强 ======================

def augment(images, mask):
    """
    随机增强：翻转、90° 旋转、亮度扰动。
    images: dict {key: (H,W) ndarray}
    mask:   (H,W) ndarray
    返回增强后的 images, mask。
    """
    if random.random() > 0.5:
        images = {k: np.fliplr(v).copy() for k, v in images.items()}
        mask = np.fliplr(mask).copy()
    if random.random() > 0.5:
        images = {k: np.flipud(v).copy() for k, v in images.items()}
        mask = np.flipud(mask).copy()
    if random.random() > 0.5:
        k = random.choice([1, 2, 3])
        images = {key: np.rot90(v, k).copy() for key, v in images.items()}
        mask = np.rot90(mask, k).copy()
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        shift = random.uniform(-0.1, 0.1)
        images = {k: v * scale + shift for k, v in images.items()}
    return images, mask


# ====================== PyTorch Dataset ======================

class LiverDataset(Dataset):
    """
    加载预处理后的 .npz 缓存，按切片索引。
    """
    def __init__(self, case_ids, train=False):
        super().__init__()
        self.train = train
        self.items = []          # list of (case_id, slice_idx)
        self.volumes = {}        # case_id -> dict of arrays

        for cid in case_ids:
            path = os.path.join(config.CACHE_DIR, f"case_{cid}.npz")
            data = np.load(path)
            self.volumes[cid] = {
                "t1in": data["t1_inphase"],
                "t1out": data["t1_outphase"],
                "t2": data["t2"],
                "mask": data["liver_mask"],
            }
            n_slices = data["t2"].shape[0]
            for s in range(n_slices):
                self.items.append((cid, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cid, s = self.items[idx]
        vol = self.volumes[cid]
        images = {k: vol[k][s].copy() for k in ("t1in", "t1out", "t2")}
        mask = vol["mask"][s].copy()

        if self.train:
            images, mask = augment(images, mask)

        out = {k: torch.from_numpy(np.ascontiguousarray(v)).float()
               for k, v in images.items()}
        out["mask"] = torch.from_numpy(np.ascontiguousarray(mask)).float()
        out["case_id"] = cid
        out["slice_idx"] = s
        return out


# ====================== 交叉验证划分 ======================

def get_fold_splits():
    """返回 list[(train_ids, val_ids)]，共 NUM_FOLDS 折。"""
    ids = np.array(config.CASE_IDS)
    kf = KFold(n_splits=config.NUM_FOLDS, shuffle=True,
               random_state=config.SEED)
    splits = []
    for train_idx, val_idx in kf.split(ids):
        splits.append((ids[train_idx].tolist(), ids[val_idx].tolist()))
    return splits
