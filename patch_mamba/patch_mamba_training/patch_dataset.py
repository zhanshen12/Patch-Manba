import ast
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EARTH_RADIUS_M = 6371000.0
KNOT_TO_MPS = 0.514444
EPS = 1e-6


# =====================================================================================
# 基础工具
# =====================================================================================
def clean_time_string(time_str):
    s = str(time_str).strip()
    s = s.replace('""', '"')
    s = s.strip('"').strip("'").strip()
    return s


def inverse_minmax(norm_value, x_min, x_max):
    norm_value = np.asarray(norm_value, dtype=np.float32)

    if np.isscalar(x_min) and np.isscalar(x_max):
        if abs(x_max - x_min) < 1e-8:
            raw = np.full_like(norm_value, fill_value=x_min, dtype=np.float32)
        else:
            raw = norm_value * (x_max - x_min) + x_min
        return np.round(raw, 6).astype(np.float32)

    x_min = np.asarray(x_min, dtype=np.float32)
    x_max = np.asarray(x_max, dtype=np.float32)
    diff = x_max - x_min

    raw = np.where(np.abs(diff) < 1e-8, x_min, norm_value * diff + x_min)
    return np.round(raw, 6).astype(np.float32)


def restore_pred_lonlat(pred_norm_xy, restore_info):
    pred_norm_xy = np.asarray(pred_norm_xy, dtype=np.float32)
    restore_info = np.asarray(restore_info, dtype=np.float32)

    if pred_norm_xy.ndim == 1:
        lon_norm = pred_norm_xy[0]
        lat_norm = pred_norm_xy[1]
        lon_min, lon_max, lat_min, lat_max = restore_info
        lon_raw = inverse_minmax(lon_norm, lon_min, lon_max)
        lat_raw = inverse_minmax(lat_norm, lat_min, lat_max)
        return np.array([lon_raw, lat_raw], dtype=np.float32)

    lon_norm = pred_norm_xy[:, 0]
    lat_norm = pred_norm_xy[:, 1]
    lon_min = restore_info[:, 0]
    lon_max = restore_info[:, 1]
    lat_min = restore_info[:, 2]
    lat_max = restore_info[:, 3]

    lon_raw = inverse_minmax(lon_norm, lon_min, lon_max)
    lat_raw = inverse_minmax(lat_norm, lat_min, lat_max)
    return np.stack([lon_raw, lat_raw], axis=1).astype(np.float32)


# =====================================================================================
# 原始轨迹读取
# =====================================================================================
def parse_track_cell_raw(cell):
    if pd.isna(cell):
        return None

    if isinstance(cell, str):
        cell = cell.strip()
        if cell == "":
            return None
        cell = cell.replace('""', '"')

    try:
        traj = ast.literal_eval(cell) if isinstance(cell, str) else cell
    except Exception as e:
        print(f"轨迹单元解析失败: {e}")
        return None

    if not isinstance(traj, (list, tuple)) or len(traj) == 0:
        return None

    parsed = []
    for point in traj:
        if not isinstance(point, (list, tuple)) or len(point) < 5:
            continue
        try:
            lon = float(point[0])
            lat = float(point[1])
            sog = float(point[2])
            cog = float(point[3])
            time_str = clean_time_string(point[4])
            ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp()
            parsed.append([lon, lat, sog, cog, ts])
        except Exception as e:
            print(f"单个轨迹点解析失败，point={point}, error={e}")
            continue

    if len(parsed) <= 1:
        return None

    arr = np.asarray(parsed, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 4])]
    return arr


def read_csv_auto_encoding(csv_path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1"]
    df = None
    used_encoding = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            used_encoding = enc
            print(f"成功读取 CSV，编码为: {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"尝试编码 {enc} 失败: {e}")

    if df is None:
        raise ValueError("CSV 文件读取失败，请检查文件路径、编码格式或文件内容。")

    df.columns = [str(c).strip() for c in df.columns]
    return df, used_encoding


def resolve_source_column(df, source_name):
    source_name = str(source_name).strip()
    actual_columns = list(df.columns)

    if source_name in actual_columns:
        return source_name
    if source_name == "radar" and "rader" in actual_columns:
        return "rader"

    raise ValueError(f"列 {source_name} 不存在。当前列名: {actual_columns}")


def load_tracks_from_csv_raw_single_source(csv_path, source_name):
    df, _ = read_csv_auto_encoding(csv_path)
    col = resolve_source_column(df, source_name)

    all_tracks = []
    for _, row in df.iterrows():
        raw_arr = parse_track_cell_raw(row[col])
        if raw_arr is not None and len(raw_arr) > 1:
            all_tracks.append(raw_arr.astype(np.float32))
    return all_tracks


# =====================================================================================
# raw-first：先算后归一化
# =====================================================================================
def get_track_restore_info(raw_arr):
    lon = raw_arr[:, 0]
    lat = raw_arr[:, 1]
    return np.array([np.min(lon), np.max(lon), np.min(lat), np.max(lat)], dtype=np.float32)


def project_point_by_sog_cog(lon_deg, lat_deg, sog_knots, cog_deg, delta_seconds):
    if delta_seconds <= 0:
        return float(lon_deg), float(lat_deg)

    speed_mps = max(float(sog_knots), 0.0) * KNOT_TO_MPS
    distance_m = speed_mps * float(delta_seconds)

    brng = math.radians(cog_deg % 360.0)
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    ang = distance_m / EARTH_RADIUS_M

    sin_lat2 = math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brng)
    sin_lat2 = min(1.0, max(-1.0, sin_lat2))
    lat2 = math.asin(sin_lat2)

    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    lon2 = (lon2 + math.pi) % (2.0 * math.pi) - math.pi
    return float(math.degrees(lon2)), float(math.degrees(lat2))


def encode_raw_point_to_feat10(raw_point, sample_start_ts, restore_info):
    lon, lat, sog, cog, ts = raw_point[:5]
    lon_min, lon_max, lat_min, lat_max = restore_info

    lon_norm = 0.0 if abs(lon_max - lon_min) < 1e-8 else (lon - lon_min) / (lon_max - lon_min)
    lat_norm = 0.0 if abs(lat_max - lat_min) < 1e-8 else (lat - lat_min) / (lat_max - lat_min)

    sog_div10 = sog / 10.0
    cog_rad = np.deg2rad(cog)
    cog_sin = np.sin(cog_rad)
    cog_cos = np.cos(cog_rad)
    rel_time_min = (ts - sample_start_ts) / 60.0

    feat = np.array(
        [lon_norm, lat_norm, sog_div10, cog_sin, cog_cos, rel_time_min, lon_min, lon_max, lat_min, lat_max],
        dtype=np.float32,
    )
    return np.round(feat, 5).astype(np.float32)


def append_interp_flag(raw_arr):
    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    return np.concatenate([raw_arr, np.zeros((len(raw_arr), 1), dtype=np.float32)], axis=1)


def sort_points6(points_arr):
    if points_arr is None or len(points_arr) == 0:
        return np.empty((0, 6), dtype=np.float32)
    points_arr = np.asarray(points_arr, dtype=np.float32)
    return points_arr[np.argsort(points_arr[:, 4])]


def build_recursive_mixed_points(observed_points6, generated_points6):
    observed_points6 = sort_points6(observed_points6)
    generated_points6 = sort_points6(generated_points6)

    if len(observed_points6) == 0 and len(generated_points6) == 0:
        return np.empty((0, 6), dtype=np.float32)
    if len(observed_points6) == 0:
        return generated_points6.astype(np.float32)
    if len(generated_points6) == 0:
        return observed_points6.astype(np.float32)

    mixed = np.concatenate([observed_points6, generated_points6], axis=0)
    return sort_points6(mixed).astype(np.float32)


def generate_future_fixed_points_from_raw(raw_arr, cut_time_ts, future_step_minutes=5, future_end_time_ts=None):
    """
    伪递归标签版：
    每个未来固定点都独立基于 <= 目标时刻 的最近真实轨迹点生成，
    不使用模型预测结果继续滚动生成标签，因此不是“真递归”。
    """
    raw_arr = np.asarray(raw_arr, dtype=np.float32)
    raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]

    if future_end_time_ts is None:
        future_end_time_ts = raw_arr[-1, 4]

    step_sec = int(future_step_minutes * 60)
    if future_end_time_ts - cut_time_ts < step_sec - EPS:
        return np.empty((0, 6), dtype=np.float32)

    target_times = np.arange(cut_time_ts + step_sec, future_end_time_ts + EPS, step_sec, dtype=np.float64)
    out = []

    for tgt_ts in target_times:
        src_idx = np.searchsorted(raw_arr[:, 4], tgt_ts, side="right") - 1
        if src_idx < 0:
            continue

        base_lon, base_lat, base_sog, base_cog, base_ts = raw_arr[src_idx]
        if abs(base_ts - tgt_ts) < EPS:
            lon_t, lat_t = base_lon, base_lat
            sog_t, cog_t = base_sog, base_cog
            interp_flag = 0.0
        else:
            lon_t, lat_t = project_point_by_sog_cog(base_lon, base_lat, base_sog, base_cog, tgt_ts - base_ts)
            sog_t, cog_t = base_sog, base_cog
            interp_flag = 1.0

        out.append([lon_t, lat_t, sog_t, cog_t, tgt_ts, interp_flag])

    if len(out) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def collect_input_patches_as_feat10(points_arr, window_start_ts, input_patch_num=12, patch_minutes=15, restore_info=None):
    patch_sec = int(patch_minutes * 60)
    all_feats = []
    all_patch_ids = []
    patch_mask = np.zeros((input_patch_num,), dtype=np.float32)

    if points_arr is None or len(points_arr) == 0:
        return np.empty((0, 10), dtype=np.float32), np.empty((0,), dtype=np.int64), patch_mask, 0

    points_arr = np.asarray(points_arr, dtype=np.float32)
    points_arr = points_arr[np.argsort(points_arr[:, 4])]

    for p in range(input_patch_num):
        left = window_start_ts + p * patch_sec
        right = left + patch_sec
        if p < input_patch_num - 1:
            mask = (points_arr[:, 4] >= left - EPS) & (points_arr[:, 4] < right - EPS)
        else:
            mask = (points_arr[:, 4] >= left - EPS) & (points_arr[:, 4] <= right + EPS)

        patch_points = points_arr[mask]
        if len(patch_points) > 0:
            patch_mask[p] = 1.0
            for point in patch_points:
                feat10 = encode_raw_point_to_feat10(raw_point=point[:5], sample_start_ts=window_start_ts, restore_info=restore_info)
                all_feats.append(feat10)
                all_patch_ids.append(p + 1)

    if len(all_feats) == 0:
        return np.empty((0, 10), dtype=np.float32), np.empty((0,), dtype=np.int64), patch_mask, 0

    data_sequence = np.stack(all_feats, axis=0).astype(np.float32)
    patch_index = np.asarray(all_patch_ids, dtype=np.int64)
    return data_sequence, patch_index, patch_mask, len(data_sequence)


# =====================================================================================
# 单步 flat 样本 Dataset / batch
# =====================================================================================
def pack_samples_to_batch(samples, pad_value=0.0):
    if len(samples) == 0:
        return {
            "data_sequence": np.empty((0, 0, 10), dtype=np.float32),
            "model_input": np.empty((0, 0, 6), dtype=np.float32),
            "sequence_mask": np.empty((0, 0), dtype=np.float32),
            "patch_index": np.empty((0, 0), dtype=np.int64),
            "patch_mask": np.empty((0, 0), dtype=np.float32),
            "label": np.empty((0, 10), dtype=np.float32),
            "model_label": np.empty((0, 5), dtype=np.float32),
            "restore_info": np.empty((0, 4), dtype=np.float32),
            "track_id": np.empty((0,), dtype=np.int64),
            "source_name": np.empty((0,), dtype=object),
        }

    N = len(samples)
    Lmax = max(s["data_sequence"].shape[0] for s in samples)
    P = len(samples[0]["patch_mask"])

    data_sequence = np.full((N, Lmax, 10), pad_value, dtype=np.float32)
    model_input = np.full((N, Lmax, 6), pad_value, dtype=np.float32)
    sequence_mask = np.zeros((N, Lmax), dtype=np.float32)
    patch_index = np.zeros((N, Lmax), dtype=np.int64)
    patch_mask = np.zeros((N, P), dtype=np.float32)
    label = np.zeros((N, 10), dtype=np.float32)
    model_label = np.zeros((N, 5), dtype=np.float32)
    restore_info = np.zeros((N, 4), dtype=np.float32)
    track_id = np.zeros((N,), dtype=np.int64)
    source_name = np.empty((N,), dtype=object)

    for i, s in enumerate(samples):
        L = s["data_sequence"].shape[0]
        if L > 0:
            data_sequence[i, :L] = s["data_sequence"]
            model_input[i, :L] = s["data_sequence"][:, :6]
            sequence_mask[i, :L] = 1.0
            patch_index[i, :L] = s["patch_index"]

        patch_mask[i] = s["patch_mask"]
        label[i] = s["label"]
        model_label[i] = s["label"][:5]
        restore_info[i] = s["restore_info"]
        track_id[i] = int(s.get("track_id", i))
        source_name[i] = str(s.get("source_name", "unknown"))

    return {
        "data_sequence": data_sequence,
        "model_input": model_input,
        "sequence_mask": sequence_mask,
        "patch_index": patch_index,
        "patch_mask": patch_mask,
        "label": label,
        "model_label": model_label,
        "restore_info": restore_info,
        "track_id": track_id,
        "source_name": source_name,
    }


class PatchForecastDataset(Dataset):
    def __init__(self, batch_data):
        self.data_sequence = torch.tensor(batch_data["data_sequence"], dtype=torch.float32)
        self.model_input = torch.tensor(batch_data["model_input"], dtype=torch.float32)
        self.sequence_mask = torch.tensor(batch_data["sequence_mask"], dtype=torch.float32)
        self.patch_index = torch.tensor(batch_data["patch_index"], dtype=torch.long)
        self.patch_mask = torch.tensor(batch_data["patch_mask"], dtype=torch.float32)
        self.label = torch.tensor(batch_data["label"], dtype=torch.float32)
        self.model_label = torch.tensor(batch_data["model_label"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.track_id = torch.tensor(batch_data.get("track_id", np.zeros((len(self.data_sequence),), dtype=np.int64)), dtype=torch.long)
        self.source_name = list(batch_data.get("source_name", ["unknown"] * len(self.data_sequence)))

    def __len__(self):
        return len(self.data_sequence)

    def __getitem__(self, idx):
        return {
            "data_sequence": self.data_sequence[idx],
            "model_input": self.model_input[idx],
            "sequence_mask": self.sequence_mask[idx],
            "patch_index": self.patch_index[idx],
            "patch_mask": self.patch_mask[idx],
            "label": self.label[idx],
            "model_label": self.model_label[idx],
            "restore_info": self.restore_info[idx],
            "track_id": self.track_id[idx],
            "source_name": self.source_name[idx],
        }


# =====================================================================================
# rollout 样本 Dataset / batch（真递归训练）
# =====================================================================================
def pack_rollout_samples_to_batch(samples):
    if len(samples) == 0:
        return {
            "observed_points6": np.empty((0, 0, 6), dtype=np.float32),
            "observed_points6_mask": np.empty((0, 0), dtype=np.float32),
            "future_points6": np.empty((0, 0, 6), dtype=np.float32),
            "future_labels": np.empty((0, 0, 10), dtype=np.float32),
            "future_model_labels": np.empty((0, 0, 5), dtype=np.float32),
            "rollout_mask": np.empty((0, 0), dtype=np.float32),
            "restore_info": np.empty((0, 4), dtype=np.float32),
            "cut_time_ts": np.empty((0,), dtype=np.float64),
            "source_name": np.empty((0,), dtype=object),
            "track_id": np.empty((0,), dtype=np.int64),
        }

    N = len(samples)
    Omax = max(s["observed_points6"].shape[0] for s in samples)
    Tmax = max(s["future_points6"].shape[0] for s in samples)

    observed_points6 = np.zeros((N, Omax, 6), dtype=np.float32)
    observed_points6_mask = np.zeros((N, Omax), dtype=np.float32)
    future_points6 = np.zeros((N, Tmax, 6), dtype=np.float32)
    future_labels = np.zeros((N, Tmax, 10), dtype=np.float32)
    future_model_labels = np.zeros((N, Tmax, 5), dtype=np.float32)
    rollout_mask = np.zeros((N, Tmax), dtype=np.float32)
    restore_info = np.zeros((N, 4), dtype=np.float32)
    cut_time_ts = np.zeros((N,), dtype=np.float64)
    source_name = np.empty((N,), dtype=object)
    track_id = np.zeros((N,), dtype=np.int64)

    for i, s in enumerate(samples):
        O = s["observed_points6"].shape[0]
        T = s["future_points6"].shape[0]
        if O > 0:
            observed_points6[i, :O] = s["observed_points6"]
            observed_points6_mask[i, :O] = 1.0
        if T > 0:
            future_points6[i, :T] = s["future_points6"]
            future_labels[i, :T] = s["future_labels"]
            future_model_labels[i, :T] = s["future_model_labels"]
            rollout_mask[i, :T] = 1.0
        restore_info[i] = s["restore_info"]
        cut_time_ts[i] = s["cut_time_ts"]
        source_name[i] = s["source_name"]
        track_id[i] = s["track_id"]

    return {
        "observed_points6": observed_points6,
        "observed_points6_mask": observed_points6_mask,
        "future_points6": future_points6,
        "future_labels": future_labels,
        "future_model_labels": future_model_labels,
        "rollout_mask": rollout_mask,
        "restore_info": restore_info,
        "cut_time_ts": cut_time_ts,
        "source_name": source_name,
        "track_id": track_id,
    }


class PatchForecastRolloutDataset(Dataset):
    def __init__(self, batch_data):
        self.observed_points6 = torch.tensor(batch_data["observed_points6"], dtype=torch.float32)
        self.observed_points6_mask = torch.tensor(batch_data["observed_points6_mask"], dtype=torch.float32)
        self.future_points6 = torch.tensor(batch_data["future_points6"], dtype=torch.float32)
        self.future_labels = torch.tensor(batch_data["future_labels"], dtype=torch.float32)
        self.future_model_labels = torch.tensor(batch_data["future_model_labels"], dtype=torch.float32)
        self.rollout_mask = torch.tensor(batch_data["rollout_mask"], dtype=torch.float32)
        self.restore_info = torch.tensor(batch_data["restore_info"], dtype=torch.float32)
        self.cut_time_ts = torch.tensor(batch_data["cut_time_ts"], dtype=torch.float64)
        self.source_name = list(batch_data["source_name"])
        self.track_id = torch.tensor(batch_data["track_id"], dtype=torch.long)

    def __len__(self):
        return len(self.future_model_labels)

    def __getitem__(self, idx):
        return {
            "observed_points6": self.observed_points6[idx],
            "observed_points6_mask": self.observed_points6_mask[idx],
            "future_points6": self.future_points6[idx],
            "future_labels": self.future_labels[idx],
            "future_model_labels": self.future_model_labels[idx],
            "rollout_mask": self.rollout_mask[idx],
            "restore_info": self.restore_info[idx],
            "cut_time_ts": self.cut_time_ts[idx],
            "track_id": self.track_id[idx],
            "source_name": self.source_name[idx],
        }


# =====================================================================================
# 核心数据集构建：伪递归版（原有训练）
# =====================================================================================
def build_patch_forecast_dataset_from_raw_tracks_pseudo(
    tracks_raw,
    source_name="unknown",
    input_patch_num=12,
    patch_minutes=15,
    strict=False,
    pad_value=0.0,
    future_step_minutes=5,
    sample_stride_minutes=5,
    min_total_input_points=1,
    max_future_steps=None,
):
    samples = []

    patch_sec = int(patch_minutes * 60)
    input_horizon_sec = input_patch_num * patch_sec
    future_step_sec = int(future_step_minutes * 60)
    stride_sec = int(sample_stride_minutes * 60)

    for track_id, raw_arr in enumerate(tracks_raw):
        if raw_arr is None or len(raw_arr) <= 1:
            continue

        raw_arr = np.asarray(raw_arr, dtype=np.float32)
        raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]

        track_start = raw_arr[0, 4]
        track_end = raw_arr[-1, 4]

        latest_base_ws = track_end - input_horizon_sec - future_step_sec
        if latest_base_ws < track_start:
            continue

        restore_info = get_track_restore_info(raw_arr)
        real_points6 = append_interp_flag(raw_arr)
        base_window_starts = np.arange(track_start, latest_base_ws + EPS, stride_sec, dtype=np.float64)

        for base_ws in base_window_starts:
            base_we = base_ws + input_horizon_sec
            observed_points6 = real_points6[real_points6[:, 4] <= base_we + EPS]

            future_points = generate_future_fixed_points_from_raw(
                raw_arr=raw_arr,
                cut_time_ts=base_we,
                future_step_minutes=future_step_minutes,
                future_end_time_ts=track_end,
            )
            if len(future_points) == 0:
                continue

            if max_future_steps is not None:
                future_points = future_points[: int(max_future_steps)]
                if len(future_points) == 0:
                    continue

            for s in range(len(future_points)):
                cur_ws = base_ws + s * future_step_sec
                cur_we = cur_ws + input_horizon_sec
                cur_label_point = future_points[s]
                prev_generated = future_points[:s] if s > 0 else np.empty((0, 6), dtype=np.float32)

                mixed_points = build_recursive_mixed_points(observed_points6=observed_points6, generated_points6=prev_generated)
                window_mask = (mixed_points[:, 4] >= cur_ws - EPS) & (mixed_points[:, 4] <= cur_we + EPS)
                window_points = mixed_points[window_mask]

                data_sequence, patch_index, patch_mask, point_count = collect_input_patches_as_feat10(
                    points_arr=window_points,
                    window_start_ts=cur_ws,
                    input_patch_num=input_patch_num,
                    patch_minutes=patch_minutes,
                    restore_info=restore_info,
                )

                if point_count < min_total_input_points:
                    continue
                if strict and np.sum(patch_mask) < input_patch_num:
                    continue

                label = encode_raw_point_to_feat10(
                    raw_point=cur_label_point[:5],
                    sample_start_ts=cur_ws,
                    restore_info=restore_info,
                )

                samples.append(
                    {
                        "source_name": str(source_name),
                        "data_sequence": data_sequence.astype(np.float32),
                        "patch_index": patch_index.astype(np.int64),
                        "patch_mask": patch_mask.astype(np.float32),
                        "label": label.astype(np.float32),
                        "restore_info": restore_info.astype(np.float32),
                        "track_id": int(track_id),
                        "sample_type": "normal" if s == 0 else "recursive",
                        "recursive_step": int(s),
                        "window_start_ts": float(cur_ws),
                        "window_end_ts": float(cur_we),
                        "future_time_ts": float(cur_label_point[4]),
                        "future_interp_flag": float(cur_label_point[5]),
                        "feedback_point_count": int(len(prev_generated)),
                        "input_point_count": int(point_count),
                    }
                )

    batch_data = pack_samples_to_batch(samples, pad_value=pad_value)
    dataset = PatchForecastDataset(batch_data)
    return samples, batch_data, dataset


# =====================================================================================
# 核心数据集构建：真递归 rollout 版
# =====================================================================================
def build_patch_rollout_dataset_from_raw_tracks(
    tracks_raw,
    source_name="unknown",
    input_patch_num=12,
    patch_minutes=15,
    strict=False,
    pad_value=0.0,
    future_step_minutes=5,
    sample_stride_minutes=5,
    min_total_input_points=1,
    max_future_steps=None,
):
    samples = []

    history_sec = int(input_patch_num * patch_minutes * 60)
    future_step_sec = int(future_step_minutes * 60)
    stride_sec = int(sample_stride_minutes * 60)

    for track_id, raw_arr in enumerate(tracks_raw):
        if raw_arr is None or len(raw_arr) <= 1:
            continue

        raw_arr = np.asarray(raw_arr, dtype=np.float32)
        raw_arr = raw_arr[np.argsort(raw_arr[:, 4])]
        real_points6 = append_interp_flag(raw_arr)
        restore_info = get_track_restore_info(raw_arr)

        track_start = raw_arr[0, 4]
        track_end = raw_arr[-1, 4]

        earliest_cut = track_start + history_sec
        latest_cut = track_end - future_step_sec
        if latest_cut < earliest_cut:
            continue

        cut_times = np.arange(earliest_cut, latest_cut + EPS, stride_sec, dtype=np.float64)

        for cut_time_ts in cut_times:
            observed_points6 = real_points6[real_points6[:, 4] <= cut_time_ts + EPS]

            future_points = generate_future_fixed_points_from_raw(
                raw_arr=raw_arr,
                cut_time_ts=cut_time_ts,
                future_step_minutes=future_step_minutes,
                future_end_time_ts=track_end,
            )
            if len(future_points) == 0:
                continue

            if max_future_steps is not None:
                future_points = future_points[: int(max_future_steps)]
                if len(future_points) == 0:
                    continue

            init_ws = cut_time_ts - history_sec
            init_we = cut_time_ts
            init_mask = (observed_points6[:, 4] >= init_ws - EPS) & (observed_points6[:, 4] <= init_we + EPS)
            init_window_points = observed_points6[init_mask]

            _, _, init_patch_mask, point_count = collect_input_patches_as_feat10(
                points_arr=init_window_points,
                window_start_ts=init_ws,
                input_patch_num=input_patch_num,
                patch_minutes=patch_minutes,
                restore_info=restore_info,
            )

            if point_count < min_total_input_points:
                continue
            if strict and np.sum(init_patch_mask) < input_patch_num:
                continue

            future_labels = []
            future_model_labels = []
            for step_idx in range(len(future_points)):
                current_cut = cut_time_ts + step_idx * future_step_sec
                current_ws = current_cut - history_sec
                label = encode_raw_point_to_feat10(
                    raw_point=future_points[step_idx][:5],
                    sample_start_ts=current_ws,
                    restore_info=restore_info,
                )
                future_labels.append(label.astype(np.float32))
                future_model_labels.append(label[:5].astype(np.float32))

            samples.append(
                {
                    "source_name": str(source_name),
                    "track_id": int(track_id),
                    "cut_time_ts": float(cut_time_ts),
                    "observed_points6": observed_points6.astype(np.float32),
                    "future_points6": future_points.astype(np.float32),
                    "future_labels": np.asarray(future_labels, dtype=np.float32),
                    "future_model_labels": np.asarray(future_model_labels, dtype=np.float32),
                    "restore_info": restore_info.astype(np.float32),
                }
            )

    batch_data = pack_rollout_samples_to_batch(samples)
    dataset = PatchForecastRolloutDataset(batch_data)
    return samples, batch_data, dataset


# =====================================================================================
# 统一入口
# =====================================================================================
def build_patch_forecast_dataset_from_raw_tracks(
    tracks_raw,
    source_name="unknown",
    input_patch_num=12,
    patch_minutes=15,
    strict=False,
    pad_value=0.0,
    future_step_minutes=5,
    sample_stride_minutes=5,
    min_total_input_points=1,
    max_future_steps=None,
    training_mode="pseudo_recursive",
):
    if training_mode == "pseudo_recursive":
        return build_patch_forecast_dataset_from_raw_tracks_pseudo(
            tracks_raw=tracks_raw,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
        )
    if training_mode == "recursive":
        return build_patch_rollout_dataset_from_raw_tracks(
            tracks_raw=tracks_raw,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
        )
    raise ValueError(f"不支持的 training_mode: {training_mode}")


def build_patch_forecast_dataset_from_csv_single_source(
    csv_path="data.csv",
    source_name="AIS",
    input_patch_num=12,
    patch_minutes=15,
    strict=False,
    pad_value=0.0,
    future_step_minutes=5,
    sample_stride_minutes=5,
    min_total_input_points=1,
    max_future_steps=None,
    training_mode="pseudo_recursive",
):
    tracks_raw = load_tracks_from_csv_raw_single_source(csv_path, source_name=source_name)
    return build_patch_forecast_dataset_from_raw_tracks(
        tracks_raw=tracks_raw,
        source_name=source_name,
        input_patch_num=input_patch_num,
        patch_minutes=patch_minutes,
        strict=strict,
        pad_value=pad_value,
        future_step_minutes=future_step_minutes,
        sample_stride_minutes=sample_stride_minutes,
        min_total_input_points=min_total_input_points,
        max_future_steps=max_future_steps,
        training_mode=training_mode,
    )


# =====================================================================================
# CSV I/O helpers（已合并自 patch_data_io.py）
# =====================================================================================
def ndarray_to_json(arr):
    arr = np.asarray(arr)
    return json.dumps(arr.tolist(), ensure_ascii=False)


def json_to_ndarray(s, dtype=np.float32):
    return np.asarray(json.loads(s), dtype=dtype)


def samples_to_dataframe(samples):
    rows = []
    for s in samples:
        rows.append(
            {
                "source_name": s["source_name"],
                "track_id": s["track_id"],
                "sample_type": s["sample_type"],
                "recursive_step": s["recursive_step"],
                "window_start_ts": s["window_start_ts"],
                "window_end_ts": s["window_end_ts"],
                "future_time_ts": s["future_time_ts"],
                "future_interp_flag": s["future_interp_flag"],
                "feedback_point_count": s["feedback_point_count"],
                "input_point_count": s["input_point_count"],
                "data_sequence_json": ndarray_to_json(np.round(s["data_sequence"], 5)),
                "patch_index_json": ndarray_to_json(s["patch_index"].astype(np.int64)),
                "patch_mask_json": ndarray_to_json(np.round(s["patch_mask"], 5)),
                "label_json": ndarray_to_json(np.round(s["label"], 5)),
                "restore_info_json": ndarray_to_json(np.round(s["restore_info"], 6)),
            }
        )
    return pd.DataFrame(rows)


def save_samples_to_csv(samples, output_csv):
    folder = os.path.dirname(output_csv)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df = samples_to_dataframe(samples)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已保存: {output_csv}，样本数 = {len(df)}")


def load_saved_samples_from_csv(saved_csv):
    df = pd.read_csv(saved_csv, encoding="utf-8-sig")
    samples = []
    for _, row in df.iterrows():
        data_sequence = json_to_ndarray(row["data_sequence_json"], dtype=np.float32)
        patch_index = json_to_ndarray(row["patch_index_json"], dtype=np.int64)
        patch_mask = json_to_ndarray(row["patch_mask_json"], dtype=np.float32)
        label = json_to_ndarray(row["label_json"], dtype=np.float32)
        restore_info = json_to_ndarray(row["restore_info_json"], dtype=np.float32)

        samples.append(
            {
                "source_name": str(row["source_name"]),
                "track_id": int(row["track_id"]),
                "sample_type": str(row["sample_type"]),
                "recursive_step": int(row["recursive_step"]),
                "window_start_ts": float(row["window_start_ts"]),
                "window_end_ts": float(row["window_end_ts"]),
                "future_time_ts": float(row["future_time_ts"]),
                "future_interp_flag": float(row["future_interp_flag"]),
                "feedback_point_count": int(row["feedback_point_count"]),
                "input_point_count": int(row["input_point_count"]),
                "data_sequence": data_sequence.astype(np.float32),
                "patch_index": patch_index.astype(np.int64),
                "patch_mask": patch_mask.astype(np.float32),
                "label": label.astype(np.float32),
                "restore_info": restore_info.astype(np.float32),
            }
        )
    return samples


def load_saved_dataset_from_csv(saved_csv, pad_value=0.0):
    samples = load_saved_samples_from_csv(saved_csv)
    batch_data = pack_samples_to_batch(samples, pad_value=pad_value)
    dataset = PatchForecastDataset(batch_data)
    return samples, batch_data, dataset


def rollout_samples_to_dataframe(samples):
    rows = []
    for s in samples:
        rows.append(
            {
                "source_name": s["source_name"],
                "track_id": s["track_id"],
                "cut_time_ts": s["cut_time_ts"],
                "observed_points6_json": ndarray_to_json(np.round(s["observed_points6"], 6)),
                "future_points6_json": ndarray_to_json(np.round(s["future_points6"], 6)),
                "future_labels_json": ndarray_to_json(np.round(s["future_labels"], 5)),
                "future_model_labels_json": ndarray_to_json(np.round(s["future_model_labels"], 5)),
                "restore_info_json": ndarray_to_json(np.round(s["restore_info"], 6)),
            }
        )
    return pd.DataFrame(rows)


def save_rollout_samples_to_csv(samples, output_csv):
    folder = os.path.dirname(output_csv)
    if folder:
        os.makedirs(folder, exist_ok=True)
    df = rollout_samples_to_dataframe(samples)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已保存 rollout: {output_csv}，样本数 = {len(df)}")


def load_rollout_samples_from_csv(saved_csv):
    df = pd.read_csv(saved_csv, encoding="utf-8-sig")
    samples = []
    for _, row in df.iterrows():
        observed_points6 = json_to_ndarray(row["observed_points6_json"], dtype=np.float32)
        future_points6 = json_to_ndarray(row["future_points6_json"], dtype=np.float32)
        future_labels = json_to_ndarray(row["future_labels_json"], dtype=np.float32)
        future_model_labels = json_to_ndarray(row["future_model_labels_json"], dtype=np.float32)
        restore_info = json_to_ndarray(row["restore_info_json"], dtype=np.float32)

        samples.append(
            {
                "source_name": str(row["source_name"]),
                "track_id": int(row["track_id"]),
                "cut_time_ts": float(row["cut_time_ts"]),
                "observed_points6": observed_points6.astype(np.float32),
                "future_points6": future_points6.astype(np.float32),
                "future_labels": future_labels.astype(np.float32),
                "future_model_labels": future_model_labels.astype(np.float32),
                "restore_info": restore_info.astype(np.float32),
            }
        )
    return samples


def load_rollout_dataset_from_csv(saved_csv):
    samples = load_rollout_samples_from_csv(saved_csv)
    batch_data = pack_rollout_samples_to_batch(samples)
    dataset = PatchForecastRolloutDataset(batch_data)
    return samples, batch_data, dataset


def build_output_csv_path(output_dir, source_name, input_patch_num, patch_minutes, future_step_minutes, training_mode="pseudo_recursive"):
    source_alias = {"AIS": "ais", "radar": "radar", "bd": "bd"}
    source_key = source_alias.get(source_name, str(source_name).lower())
    suffix = "pseudo" if training_mode == "pseudo_recursive" else "recursive"
    return os.path.join(output_dir, f"{source_key}_{input_patch_num}batch_{patch_minutes}min_{future_step_minutes}min_{suffix}.csv")


def default_window_configs():
    return [
        {"name": "win15_12x15", "input_patch_num": 12, "patch_minutes": 15},
        {"name": "win10_18x10", "input_patch_num": 18, "patch_minutes": 10},
        {"name": "win20_9x20", "input_patch_num": 9, "patch_minutes": 20},
        {"name": "win30_6x30", "input_patch_num": 6, "patch_minutes": 30},
        {"name": "win25_7x25", "input_patch_num": 7, "patch_minutes": 25},
    ]


def build_and_save_source_multiscale(
    csv_path="data.csv",
    output_dir="prebuilt_source_csv",
    source_name="AIS",
    window_configs=None,
    strict=False,
    pad_value=0.0,
    future_step_minutes=5,
    sample_stride_minutes=5,
    min_total_input_points=1,
    max_future_steps=12,
    training_mode="pseudo_recursive",
):
    if window_configs is None:
        window_configs = default_window_configs()

    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for cfg in window_configs:
        name = str(cfg["name"])
        input_patch_num = int(cfg["input_patch_num"])
        patch_minutes = int(cfg["patch_minutes"])

        print("\n" + "=" * 100)
        print(f"处理 source={source_name} | config={name} | {input_patch_num}x{patch_minutes} | mode={training_mode}")
        print("=" * 100)

        samples, batch_data, _ = build_patch_forecast_dataset_from_csv_single_source(
            csv_path=csv_path,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            strict=strict,
            pad_value=pad_value,
            future_step_minutes=future_step_minutes,
            sample_stride_minutes=sample_stride_minutes,
            min_total_input_points=min_total_input_points,
            max_future_steps=max_future_steps,
            training_mode=training_mode,
        )

        output_csv = build_output_csv_path(
            output_dir=output_dir,
            source_name=source_name,
            input_patch_num=input_patch_num,
            patch_minutes=patch_minutes,
            future_step_minutes=future_step_minutes,
            training_mode=training_mode,
        )
        if training_mode == "pseudo_recursive":
            save_samples_to_csv(samples, output_csv)
            normal_cnt = sum(s["sample_type"] == "normal" for s in samples)
            recursive_cnt = sum(s["sample_type"] == "recursive" for s in samples)
        else:
            save_rollout_samples_to_csv(samples, output_csv)
            normal_cnt = len(samples)
            recursive_cnt = len(samples)

        summary_rows.append(
            {
                "source_name": source_name,
                "config_name": name,
                "training_mode": training_mode,
                "input_patch_num": input_patch_num,
                "patch_minutes": patch_minutes,
                "history_minutes": input_patch_num * patch_minutes,
                "future_step_minutes": future_step_minutes,
                "max_future_steps": max_future_steps,
                "output_csv": output_csv,
                "sample_count": len(samples),
                "normal_count": normal_cnt,
                "recursive_count": recursive_cnt,
            }
        )

        print(f"完成: output={output_csv} | samples={len(samples)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, f"summary_{str(source_name).lower()}_{training_mode}.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n汇总信息已保存: {summary_csv}")
    return summary_df


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    summary_df = build_and_save_source_multiscale(
        csv_path="data41.csv",
        output_dir="prebuilt_source_csv",
        source_name="AIS",
        window_configs=default_window_configs(),
        strict=False,
        pad_value=0.0,
        future_step_minutes=5,
        sample_stride_minutes=5,
        min_total_input_points=1,
        max_future_steps=12,
        training_mode="pseudo_recursive",
    )

    print("\n汇总结果:")
    print(summary_df)
