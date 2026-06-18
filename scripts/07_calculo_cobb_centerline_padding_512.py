import csv
import os
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, savgol_filter

from config import get_dataset_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIGURACOES
# =========================

CENTERLINE_SUBDIR = "unet_baseline_2000_padding_512_centerline_100"
OUTPUT_SUBDIR = (
    Path("baseline_2000")
    / "multicobb"
    / "active"
    / "unet_baseline_2000_padding_512_multicobb_100_no_centerline_smooth"
)
CENTERLINE_SUBDIR_OVERRIDE = os.environ.get("SPINAL_COBB_CENTERLINE_SUBDIR")
OUTPUT_SUBDIR_OVERRIDE = os.environ.get("SPINAL_COBB_OUTPUT_SUBDIR")

DATASET_DIR = get_dataset_dir()

PROCESSED_DIR = DATASET_DIR / "processed_padding_512"

CENTERLINE_SUBDIR = CENTERLINE_SUBDIR_OVERRIDE or CENTERLINE_SUBDIR
OUTPUT_SUBDIR = Path(OUTPUT_SUBDIR_OVERRIDE) if OUTPUT_SUBDIR_OVERRIDE else OUTPUT_SUBDIR

CENTERLINE_DIR = PROCESSED_DIR / "centerlines" / CENTERLINE_SUBDIR

OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTLIERS_VISUAL_DIR = OUTPUT_DIR / "top_outliers_visual"
OUTLIERS_VISUAL_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "cobb_results_multicobb_with_gt.csv"
OUTLIERS_CSV_PATH = OUTPUT_DIR / "top_outliers_multicobb.csv"
METRICS_CSV_PATH = OUTPUT_DIR / "metrics_multicobb.csv"

COBB_GT_PATH = DATASET_DIR / "Cobb_spinal-AI2024-train_gt.txt"

IMAGE_SIZE = 512

SMOOTH_WINDOW = 1
SMOOTH_POLYORDER = 3

ANGLE_SMOOTH_WINDOW = 21
ANGLE_SMOOTH_POLYORDER = 3

TRIM_RATIO = 0.07

MIN_POINT_DISTANCE_GLOBAL = 80
MIN_POINT_DISTANCE_REGION = 50
MIN_REGION_POINTS = 12

MAX_ABS_ANGLE = 80

LOW_PERCENTILE = 3
HIGH_PERCENTILE = 97
CANDIDATE_LIMIT = 25

PT_CURVATURE_SUPPORT_START = 0.07
PT_CURVATURE_SUPPORT_END = 0.40
PT_CURVATURE_PROMINENCE = 2.0
PT_UNSUPPORTED_TARGET_SCALE = 0.55

TL_CURVATURE_SUPPORT_START = 0.60
TL_CURVATURE_SUPPORT_END = 0.98
TL_LOW_AMPLITUDE_START = 0.70
TL_LOW_AMPLITUDE_LIMIT = 10.0
TL_SECONDARY_MIN_COBB = 15.0
TL_UNSUPPORTED_TARGET_SCALE = 0.75

LOCAL_ANGLE_WINDOW = 11
LOCAL_STD_LIMIT = 8.0
LOCAL_STD_PENALTY = 0.08
CLOSE_DISTANCE_LIMIT = 90
CLOSE_DISTANCE_PENALTY = 0.05

REGION_EXPANSION_STEP = 0.05
MAX_REGION_EXPANSION = 0.15

TANGENT_LENGTH = 90

TOP_N_OUTLIERS = 20

TOLERANCES = [5, 10, 15]

# Regioes iniciais por posicao vertical normalizada na centerline aparada.
# Sao sobrepostas de proposito para reduzir cortes rigidos entre curvas.
# O PT ignora os primeiros 7% para evitar tangentes instaveis na borda superior.
REGIONS = [
    {
        "name": "PT",
        "label": "PT",
        "gt_key": "PT",
        "start": 0.07,
        "end": 0.42,
        "top_color": "tab:red",
        "bottom_color": "tab:orange",
        "band_color": "tab:red",
    },
    {
        "name": "MT",
        "label": "MT",
        "gt_key": "MT",
        "start": 0.05,
        "end": 0.95,
        "top_color": "tab:blue",
        "bottom_color": "tab:cyan",
        "band_color": "tab:blue",
    },
    {
        "name": "TL_L",
        "label": "TL/L",
        "gt_key": "TL_L",
        "start": 0.35,
        "end": 1.00,
        "top_color": "tab:purple",
        "bottom_color": "tab:green",
        "band_color": "tab:green",
    },
]


# =========================
# FUNCOES
# =========================

def load_cobb_ground_truth(path):
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro de GT nao encontrado: {path}")

    cobb_gt = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 4:
                raise ValueError(f"Linha invalida no GT ({line_number}): {line}")

            file_name, pt, mt, tl_l = parts
            file_id = Path(file_name).stem

            cobb_gt[file_id] = {
                "PT": float(pt),
                "MT": float(mt),
                "TL_L": float(tl_l),
            }

    return cobb_gt


def make_odd(value):
    value = int(value)

    if value % 2 == 0:
        value -= 1

    return max(value, 3)


def load_centerline_points(centerline_path):
    if not centerline_path.exists():
        raise FileNotFoundError(f"Centerline nao encontrada: {centerline_path}")

    img = Image.open(centerline_path).convert("L")
    arr = np.array(img)

    ys, xs = np.where(arr > 0)

    if len(xs) == 0:
        return np.array([]), np.array([])

    unique_ys = np.unique(ys)

    x_points = []
    y_points = []

    for y in unique_ys:
        x_values = xs[ys == y]

        if len(x_values) == 0:
            continue

        x_points.append(np.mean(x_values))
        y_points.append(y)

    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)

    order = np.argsort(y_points)

    return x_points[order], y_points[order]


def smooth_centerline(xs, ys):
    if len(xs) < 7:
        return xs, ys

    window = make_odd(min(SMOOTH_WINDOW, len(xs)))

    if window <= SMOOTH_POLYORDER:
        return xs, ys

    xs_smooth = savgol_filter(
        xs,
        window_length=window,
        polyorder=SMOOTH_POLYORDER,
    )

    return xs_smooth, ys


def trim_centerline(xs, ys):
    n = len(xs)

    start = int(n * TRIM_RATIO)
    end = int(n * (1 - TRIM_RATIO))

    if end <= start:
        return xs, ys

    return xs[start:end], ys[start:end]


def calculate_angles_by_arc_length(xs, ys):
    dx = np.gradient(xs)
    dy = np.gradient(ys)

    ds = np.sqrt(dx ** 2 + dy ** 2)
    ds[ds == 0] = 1e-6

    dx_ds = dx / ds
    dy_ds = dy / ds

    angles = np.degrees(np.arctan2(dx_ds, dy_ds))
    angles = ((angles + 90) % 180) - 90

    if len(angles) >= 7:
        window = make_odd(min(ANGLE_SMOOTH_WINDOW, len(angles)))

        if window > ANGLE_SMOOTH_POLYORDER:
            angles = savgol_filter(
                angles,
                window_length=window,
                polyorder=ANGLE_SMOOTH_POLYORDER,
            )

    return angles


def angular_difference(angle1, angle2):
    diff = abs(angle2 - angle1)

    if diff > 90:
        diff = 180 - diff

    return abs(diff)


def get_local_angle_stats(angles, idx, window=LOCAL_ANGLE_WINDOW):
    n = len(angles)
    half = window // 2

    start = max(0, idx - half)
    end = min(n, idx + half + 1)

    local_values = angles[start:end]

    if len(local_values) == 0:
        return angles[idx], 0.0

    local_mean = float(np.mean(local_values))
    local_std = float(np.std(local_values))

    return local_mean, local_std


def get_effective_angle(angles, idx):
    raw_angle = float(angles[idx])
    local_mean, local_std = get_local_angle_stats(angles, idx)

    if local_std > LOCAL_STD_LIMIT:
        effective_angle = 0.75 * raw_angle + 0.25 * local_mean
    else:
        effective_angle = raw_angle

    return effective_angle, local_mean, local_std


def get_region_indices(ys, region):
    if len(ys) == 0:
        return np.array([], dtype=int), region["start"], region["end"]

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    y_span = y_max - y_min

    if y_span <= 0:
        return np.array([], dtype=int), region["start"], region["end"]

    normalized_y = (ys - y_min) / y_span

    expansion = 0.0

    while expansion <= MAX_REGION_EXPANSION + 1e-9:
        start = max(0.0, region["start"] - expansion)
        end = min(1.0, region["end"] + expansion)

        indices = np.where(
            (normalized_y >= start) & (normalized_y <= end)
        )[0]

        if len(indices) >= MIN_REGION_POINTS:
            return indices.astype(int), start, end

        expansion += REGION_EXPANSION_STEP

    return indices.astype(int), start, end


def reorder_extreme_pair(i, j, best_data, ys):
    if ys[i] <= ys[j]:
        return i, j, best_data

    reordered_data = {
        "angle_i_eff": best_data["angle_j_eff"],
        "angle_j_eff": best_data["angle_i_eff"],
        "angle_i_raw": best_data["angle_j_raw"],
        "angle_j_raw": best_data["angle_i_raw"],
        "angle_i_local": best_data["angle_j_local"],
        "angle_j_local": best_data["angle_i_local"],
        "std_i": best_data["std_j"],
        "std_j": best_data["std_i"],
        "cobb_eff": best_data["cobb_eff"],
        "cobb_raw": best_data["cobb_raw"],
        "score": best_data["score"],
        "y_distance": best_data["y_distance"],
    }

    return j, i, reordered_data


def build_extremes_result(i, j, best_data, ys):
    i, j, best_data = reorder_extreme_pair(i, j, best_data, ys)

    return {
        "idx_top": int(i),
        "idx_bottom": int(j),
        "angle_top": best_data["angle_i_eff"],
        "angle_bottom": best_data["angle_j_eff"],
        "angle_top_raw": best_data["angle_i_raw"],
        "angle_bottom_raw": best_data["angle_j_raw"],
        "angle_top_local": best_data["angle_i_local"],
        "angle_bottom_local": best_data["angle_j_local"],
        "angle_std_top": best_data["std_i"],
        "angle_std_bottom": best_data["std_j"],
        "cobb": best_data["cobb_eff"],
        "cobb_raw": best_data["cobb_raw"],
        "selection_score": best_data["score"],
        "y_distance": best_data["y_distance"],
    }


def calculate_pair_data(angles, ys, i, j):
    y_distance = abs(ys[j] - ys[i])

    angle_i_eff, angle_i_local, std_i = get_effective_angle(angles, i)
    angle_j_eff, angle_j_local, std_j = get_effective_angle(angles, j)

    cobb_eff = angular_difference(angle_i_eff, angle_j_eff)
    cobb_raw = angular_difference(angles[i], angles[j])

    std_penalty = LOCAL_STD_PENALTY * (std_i + std_j)
    distance_penalty = 0.0

    if y_distance < CLOSE_DISTANCE_LIMIT:
        distance_penalty = (
            CLOSE_DISTANCE_PENALTY * (CLOSE_DISTANCE_LIMIT - y_distance)
        )

    score = cobb_eff - std_penalty - distance_penalty

    return {
        "angle_i_eff": angle_i_eff,
        "angle_j_eff": angle_j_eff,
        "angle_i_raw": float(angles[i]),
        "angle_j_raw": float(angles[j]),
        "angle_i_local": angle_i_local,
        "angle_j_local": angle_j_local,
        "std_i": std_i,
        "std_j": std_j,
        "cobb_eff": cobb_eff,
        "cobb_raw": cobb_raw,
        "score": score,
        "y_distance": y_distance,
    }


def find_cobb_extremes(
    xs,
    ys,
    angles,
    candidate_indices=None,
    min_point_distance=MIN_POINT_DISTANCE_GLOBAL,
):
    if candidate_indices is None:
        candidate_indices = np.arange(len(angles))

    candidate_indices = np.array(candidate_indices, dtype=int)

    if len(candidate_indices) < MIN_REGION_POINTS:
        return None

    angle_mask = np.abs(angles[candidate_indices]) <= MAX_ABS_ANGLE
    valid_indices = candidate_indices[angle_mask]

    if len(valid_indices) < MIN_REGION_POINTS:
        valid_indices = candidate_indices

    valid_angles = angles[valid_indices]

    min_target = np.percentile(valid_angles, LOW_PERCENTILE)
    max_target = np.percentile(valid_angles, HIGH_PERCENTILE)

    min_candidates = valid_indices[
        np.argsort(np.abs(angles[valid_indices] - min_target))
    ]

    max_candidates = valid_indices[
        np.argsort(np.abs(angles[valid_indices] - max_target))
    ]

    best_pair = None
    best_score = -1e9
    best_data = None

    for i in min_candidates[:CANDIDATE_LIMIT]:
        for j in max_candidates[:CANDIDATE_LIMIT]:
            if i == j:
                continue

            y_distance = abs(ys[j] - ys[i])

            if y_distance < min_point_distance:
                continue

            pair_data = calculate_pair_data(angles, ys, i, j)
            score = pair_data["score"]

            if score > best_score:
                best_score = score
                best_pair = (i, j)
                best_data = pair_data

    if best_pair is None:
        return None

    i, j = best_pair

    return build_extremes_result(i, j, best_data, ys)


def find_cobb_extremes_near_relative_cobb(
    ys,
    angles,
    candidate_indices,
    min_point_distance,
    target_scale,
):
    candidate_indices = np.array(candidate_indices, dtype=int)

    if len(candidate_indices) < MIN_REGION_POINTS:
        return None

    pairs = []

    for pos_i, i in enumerate(candidate_indices):
        for j in candidate_indices[pos_i + 1:]:
            y_distance = abs(ys[j] - ys[i])

            if y_distance < min_point_distance:
                continue

            pair_data = calculate_pair_data(angles, ys, i, j)
            pairs.append((i, j, pair_data))

    if len(pairs) == 0:
        return None

    max_cobb = max(pair_data["cobb_eff"] for _, _, pair_data in pairs)
    target_cobb = max_cobb * target_scale

    i, j, best_data = min(
        pairs,
        key=lambda item: (
            abs(item[2]["cobb_eff"] - target_cobb),
            -item[2]["score"],
        ),
    )

    best_data = dict(best_data)
    best_data["score"] = best_data["score"] - abs(best_data["cobb_eff"] - target_cobb)

    return build_extremes_result(i, j, best_data, ys)


def has_pt_curvature_support(ys, angles):
    if len(ys) == 0:
        return False

    y_span = float(np.max(ys) - np.min(ys))

    if y_span <= 0:
        return False

    normalized_y = (ys - np.min(ys)) / y_span

    peak_indices, _ = find_peaks(
        angles,
        prominence=PT_CURVATURE_PROMINENCE,
        distance=8,
    )
    trough_indices, _ = find_peaks(
        -angles,
        prominence=PT_CURVATURE_PROMINENCE,
        distance=8,
    )

    landmark_indices = np.concatenate([peak_indices, trough_indices])

    if len(landmark_indices) == 0:
        return False

    landmark_positions = normalized_y[landmark_indices]

    return bool(np.any(
        (landmark_positions >= PT_CURVATURE_SUPPORT_START)
        & (landmark_positions <= PT_CURVATURE_SUPPORT_END)
    ))


def has_tl_lower_curvature_support(ys, angles):
    if len(ys) == 0:
        return False

    y_span = float(np.max(ys) - np.min(ys))

    if y_span <= 0:
        return False

    normalized_y = (ys - np.min(ys)) / y_span

    peak_indices, _ = find_peaks(
        angles,
        prominence=PT_CURVATURE_PROMINENCE,
        distance=8,
    )
    trough_indices, _ = find_peaks(
        -angles,
        prominence=PT_CURVATURE_PROMINENCE,
        distance=8,
    )

    landmark_indices = np.concatenate([peak_indices, trough_indices])

    if len(landmark_indices) == 0:
        return False

    landmark_positions = normalized_y[landmark_indices]

    return bool(np.any(
        (landmark_positions >= TL_CURVATURE_SUPPORT_START)
        & (landmark_positions <= TL_CURVATURE_SUPPORT_END)
    ))


def calculate_lower_lateral_amplitude(xs, ys):
    if len(xs) < 3 or len(ys) < 3:
        return 0.0

    y_span = float(np.max(ys) - np.min(ys))

    if y_span <= 0:
        return 0.0

    normalized_y = (ys - np.min(ys)) / y_span

    lower_indices = np.where(normalized_y >= TL_LOW_AMPLITUDE_START)[0]

    if len(lower_indices) == 0:
        return 0.0

    trend = np.polyval(np.polyfit(ys, xs, 1), ys)
    residuals = xs - trend
    lower_residuals = residuals[lower_indices]

    return float(np.max(lower_residuals) - np.min(lower_residuals))


def calculate_regional_cobbs(xs, ys, angles):
    regional = {}
    pt_has_curvature_support = has_pt_curvature_support(ys, angles)
    tl_has_lower_curvature_support = has_tl_lower_curvature_support(ys, angles)
    tl_lower_lateral_amplitude = calculate_lower_lateral_amplitude(xs, ys)

    for region in REGIONS:
        indices, start, end = get_region_indices(ys, region)

        extremes = find_cobb_extremes(
            xs,
            ys,
            angles,
            candidate_indices=indices,
            min_point_distance=MIN_POINT_DISTANCE_REGION,
        )

        status = "ok" if extremes is not None else "no_valid_pair"

        if (
            region["name"] == "PT"
            and extremes is not None
            and not pt_has_curvature_support
        ):
            secondary_extremes = find_cobb_extremes_near_relative_cobb(
                ys,
                angles,
                indices,
                min_point_distance=MIN_POINT_DISTANCE_REGION,
                target_scale=PT_UNSUPPORTED_TARGET_SCALE,
            )

            if secondary_extremes is not None:
                extremes = secondary_extremes
                status = "ok_secondary_no_upper_curvature"

        if (
            region["name"] == "TL_L"
            and extremes is not None
            and extremes["cobb"] >= TL_SECONDARY_MIN_COBB
            and not tl_has_lower_curvature_support
            and tl_lower_lateral_amplitude <= TL_LOW_AMPLITUDE_LIMIT
        ):
            secondary_extremes = find_cobb_extremes_near_relative_cobb(
                ys,
                angles,
                indices,
                min_point_distance=MIN_POINT_DISTANCE_REGION,
                target_scale=TL_UNSUPPORTED_TARGET_SCALE,
            )

            if secondary_extremes is not None:
                extremes = secondary_extremes
                status = "ok_secondary_low_lower_curvature"

        regional[region["name"]] = {
            "region": region,
            "indices": indices,
            "start": start,
            "end": end,
            "extremes": extremes,
            "status": status,
        }

    return regional


def draw_tangent(ax, x, y, angle_deg, color):
    angle_rad = np.radians(angle_deg)

    dx = np.sin(angle_rad) * TANGENT_LENGTH
    dy = np.cos(angle_rad) * TANGENT_LENGTH

    x1 = x - dx
    x2 = x + dx
    y1 = y - dy
    y2 = y + dy

    ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)


def save_visualization(
    file_id,
    xs_smooth,
    ys_smooth,
    xs_trim,
    ys_trim,
    regional_cobbs,
    gt=None,
):
    output_path = OUTPUT_DIR / f"{file_id}_multicobb.png"

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE)), cmap="gray")

    ax.plot(xs_smooth, ys_smooth, color="cyan", linewidth=2)
    ax.plot(xs_trim, ys_trim, color="white", linestyle=":", linewidth=1)

    y_min = float(np.min(ys_trim))
    y_max = float(np.max(ys_trim))
    y_span = y_max - y_min

    title_parts = []

    for region in REGIONS:
        name = region["name"]
        data = regional_cobbs[name]
        extremes = data["extremes"]

        band_y1 = y_min + data["start"] * y_span
        band_y2 = y_min + data["end"] * y_span

        ax.axhspan(
            band_y1,
            band_y2,
            color=region["band_color"],
            alpha=0.05,
        )

        if extremes is None:
            title_parts.append(f"{region['label']} NA")
            continue

        idx_top = extremes["idx_top"]
        idx_bottom = extremes["idx_bottom"]

        draw_tangent(
            ax,
            xs_trim[idx_top],
            ys_trim[idx_top],
            extremes["angle_top"],
            color=region["top_color"],
        )

        draw_tangent(
            ax,
            xs_trim[idx_bottom],
            ys_trim[idx_bottom],
            extremes["angle_bottom"],
            color=region["bottom_color"],
        )

        ax.scatter(
            [xs_trim[idx_top], xs_trim[idx_bottom]],
            [ys_trim[idx_top], ys_trim[idx_bottom]],
            color=[region["top_color"], region["bottom_color"]],
            s=35,
        )

        if gt is not None:
            title_parts.append(
                f"{region['label']} {extremes['cobb']:.1f}/{gt[region['gt_key']]:.1f}"
            )
        else:
            title_parts.append(f"{region['label']} {extremes['cobb']:.1f}")

    ax.set_title(f"{file_id} | " + " | ".join(title_parts), fontsize=10)
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def calculate_metrics(values):
    values = np.array(values, dtype=float)

    mae = np.mean(np.abs(values))
    rmse = np.sqrt(np.mean(values ** 2))

    return mae, rmse


def calculate_symmetric_smape_value(estimated, real):
    denominator = abs(estimated) + abs(real)

    if denominator == 0:
        return 0.0

    return 2.0 * abs(estimated - real) / denominator * 100.0


def calculate_mean(values):
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def calculate_benchmark_smape_value(estimated_values, real_values):
    estimated_values = np.array(estimated_values, dtype=float)
    real_values = np.array(real_values, dtype=float)
    denominator = float(np.sum(np.abs(estimated_values) + np.abs(real_values)))

    if denominator == 0:
        return 0.0

    numerator = float(np.sum(np.abs(estimated_values - real_values)))
    return 200.0 * numerator / denominator


def calculate_tolerance_accuracy(errors, tolerances=TOLERANCES):
    errors = np.array(errors, dtype=float)
    abs_errors = np.abs(errors)

    results = {}

    for tolerance in tolerances:
        results[tolerance] = np.mean(abs_errors <= tolerance) * 100

    return results


def empty_region_values(region_name):
    return {
        f"cobb_estimated_{region_name}": "",
        f"cobb_raw_{region_name}": "",
        f"cobb_real_{region_name}": "",
        f"error_{region_name}": "",
        f"abs_error_{region_name}": "",
        f"symmetric_smape_{region_name}": "",
        f"within_5_deg_{region_name}": "",
        f"within_10_deg_{region_name}": "",
        f"within_15_deg_{region_name}": "",
        f"angle_top_{region_name}": "",
        f"angle_bottom_{region_name}": "",
        f"angle_std_top_{region_name}": "",
        f"angle_std_bottom_{region_name}": "",
        f"idx_top_{region_name}": "",
        f"idx_bottom_{region_name}": "",
        f"y_distance_{region_name}": "",
        f"selection_score_{region_name}": "",
        f"region_start_{region_name}": "",
        f"region_end_{region_name}": "",
        f"status_{region_name}": "no_gt_or_no_valid_pair",
    }


def add_region_values(row, region, data, gt, errors_by_region, symmetric_smape_by_region):
    region_name = region["name"]
    gt_key = region["gt_key"]
    extremes = data["extremes"]

    row.update(empty_region_values(region_name))

    row[f"region_start_{region_name}"] = round(data["start"], 3)
    row[f"region_end_{region_name}"] = round(data["end"], 3)
    row[f"status_{region_name}"] = data["status"]

    if extremes is None:
        return None

    cobb_estimated = extremes["cobb"]
    row[f"cobb_estimated_{region_name}"] = round(cobb_estimated, 2)
    row[f"cobb_raw_{region_name}"] = round(extremes["cobb_raw"], 2)
    row[f"angle_top_{region_name}"] = round(extremes["angle_top"], 2)
    row[f"angle_bottom_{region_name}"] = round(extremes["angle_bottom"], 2)
    row[f"angle_std_top_{region_name}"] = round(extremes["angle_std_top"], 2)
    row[f"angle_std_bottom_{region_name}"] = round(extremes["angle_std_bottom"], 2)
    row[f"idx_top_{region_name}"] = extremes["idx_top"]
    row[f"idx_bottom_{region_name}"] = extremes["idx_bottom"]
    row[f"y_distance_{region_name}"] = round(extremes["y_distance"], 2)
    row[f"selection_score_{region_name}"] = round(extremes["selection_score"], 2)

    if gt is None:
        return None

    cobb_real = gt[gt_key]
    error = cobb_estimated - cobb_real
    abs_error = abs(error)
    symmetric_smape = calculate_symmetric_smape_value(cobb_estimated, cobb_real)

    row[f"cobb_real_{region_name}"] = round(cobb_real, 2)
    row[f"error_{region_name}"] = round(error, 2)
    row[f"abs_error_{region_name}"] = round(abs_error, 2)
    row[f"symmetric_smape_{region_name}"] = round(symmetric_smape, 2)
    row[f"within_5_deg_{region_name}"] = abs_error <= 5
    row[f"within_10_deg_{region_name}"] = abs_error <= 10
    row[f"within_15_deg_{region_name}"] = abs_error <= 15

    errors_by_region[region_name].append(error)
    symmetric_smape_by_region[region_name].append(symmetric_smape)

    return {
        "region": region_name,
        "estimated": cobb_estimated,
        "real": cobb_real,
        "error": error,
        "abs_error": abs_error,
        "symmetric_smape": symmetric_smape,
    }


def build_fieldnames():
    fieldnames = [
        "file_id",
        "file_id_gt",
        "cobb_global_estimated",
        "cobb_global_raw",
        "error_global_vs_MT",
        "abs_error_global_vs_MT",
        "within_5_deg_global_vs_MT",
        "within_10_deg_global_vs_MT",
        "within_15_deg_global_vs_MT",
    ]

    for region in REGIONS:
        region_name = region["name"]
        fieldnames.extend([
            f"cobb_estimated_{region_name}",
            f"cobb_raw_{region_name}",
            f"cobb_real_{region_name}",
            f"error_{region_name}",
            f"abs_error_{region_name}",
            f"symmetric_smape_{region_name}",
            f"within_5_deg_{region_name}",
            f"within_10_deg_{region_name}",
            f"within_15_deg_{region_name}",
            f"angle_top_{region_name}",
            f"angle_bottom_{region_name}",
            f"angle_std_top_{region_name}",
            f"angle_std_bottom_{region_name}",
            f"idx_top_{region_name}",
            f"idx_bottom_{region_name}",
            f"y_distance_{region_name}",
            f"selection_score_{region_name}",
            f"region_start_{region_name}",
            f"region_end_{region_name}",
            f"status_{region_name}",
        ])

    fieldnames.extend([
        "mean_abs_error_3_cobb",
        "max_abs_error_3_cobb",
        "mean_symmetric_smape_3_cobb",
        "benchmark_smape_3_cobb",
        "max_cobb_estimated",
        "max_cobb_real",
        "error_max_cobb",
        "abs_error_max_cobb",
        "within_5_deg_max_cobb",
        "within_10_deg_max_cobb",
        "within_15_deg_max_cobb",
        "visualization_path",
    ])

    return fieldnames


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_summary(metric_rows):
    fieldnames = [
        "target",
        "n",
        "mae",
        "rmse",
        "symmetric_smape",
        "benchmark_smape",
        "within_5_deg_percent",
        "within_10_deg_percent",
        "within_15_deg_percent",
        "within_5_percent",
        "within_10_percent",
        "within_15_percent",
    ]

    save_csv(METRICS_CSV_PATH, metric_rows, fieldnames)


def save_top_outliers(results):
    results_with_error = [
        row for row in results
        if row["mean_abs_error_3_cobb"] != ""
    ]

    results_with_error = sorted(
        results_with_error,
        key=lambda row: float(row["mean_abs_error_3_cobb"]),
        reverse=True,
    )

    top_outliers = results_with_error[:TOP_N_OUTLIERS]

    outlier_fieldnames = [
        "file_id",
        "mean_abs_error_3_cobb",
        "max_abs_error_3_cobb",
        "abs_error_PT",
        "abs_error_MT",
        "abs_error_TL_L",
        "cobb_estimated_PT",
        "cobb_estimated_MT",
        "cobb_estimated_TL_L",
        "cobb_real_PT",
        "cobb_real_MT",
        "cobb_real_TL_L",
        "visualization_path",
    ]

    top_outliers_clean = []

    for row in top_outliers:
        clean_row = {}

        for field in outlier_fieldnames:
            clean_row[field] = row.get(field, "")

        top_outliers_clean.append(clean_row)

    save_csv(OUTLIERS_CSV_PATH, top_outliers_clean, outlier_fieldnames)

    for row in top_outliers:
        source_path = Path(row["visualization_path"])

        if source_path.exists():
            destination_path = OUTLIERS_VISUAL_DIR / source_path.name
            shutil.copy2(source_path, destination_path)

    return top_outliers


def file_id_for_gt(file_id_raw):
    parts = file_id_raw.split("_")

    if len(parts) >= 2 and parts[0].isdigit():
        return "_".join(parts[1:])

    return file_id_raw


def format_optional(value):
    if value == "":
        return "NA"

    return f"{float(value):.2f}"


def format_signed_optional(value):
    if value == "":
        return "NA"

    return f"{float(value):+.2f}"


def print_image_result(image_number, total_images, file_id, row, gt):
    print(f"\n[{image_number:03d}/{total_images:03d}] {file_id}")

    if gt is None:
        print("  Cobbs regionais calculados | sem GT")
        return

    print("  Regiao  Estimado  Real     Desvio   |Desvio|  Estado")

    labels = {
        "PT": "PT",
        "MT": "MT",
        "TL_L": "TL/L",
    }

    for region in REGIONS:
        region_name = region["name"]
        label = labels[region_name]

        estimated = row[f"cobb_estimated_{region_name}"]
        real = row[f"cobb_real_{region_name}"]
        error = row[f"error_{region_name}"]
        abs_error = row[f"abs_error_{region_name}"]
        status = row[f"status_{region_name}"]

        print(
            f"  {label:<6} "
            f"{format_optional(estimated):>8}  "
            f"{format_optional(real):>7}  "
            f"{format_signed_optional(error):>8}  "
            f"{format_optional(abs_error):>8}  "
            f"{status}"
        )

    print(
        f"  MAE 3 Cobb: {format_optional(row['mean_abs_error_3_cobb'])} | "
        f"Maior desvio: {format_optional(row['max_abs_error_3_cobb'])}"
    )


def main():
    if not CENTERLINE_DIR.exists():
        raise FileNotFoundError(f"Pasta de centerlines nao encontrada: {CENTERLINE_DIR}")

    cobb_gt = load_cobb_ground_truth(COBB_GT_PATH)
    centerline_files = sorted(CENTERLINE_DIR.glob("*_centerline.png"))

    if len(centerline_files) == 0:
        raise FileNotFoundError(f"Nenhuma centerline encontrada em: {CENTERLINE_DIR}")

    results = []
    errors_by_region = {region["name"]: [] for region in REGIONS}
    symmetric_smape_by_region = {region["name"]: [] for region in REGIONS}
    errors_all_regions = []
    symmetric_smape_all_regions = []
    benchmark_smape_3_cobb_values = []
    errors_global_mt = []
    symmetric_smape_global_mt = []
    errors_max_cobb = []
    symmetric_smape_max_cobb = []

    print("\n===== CALCULO MULTI-COBB REGIONAL - CENTERLINE PADDING 512 =====")
    print(f"Pasta centerlines: {CENTERLINE_DIR}")
    print(f"Centerlines encontradas: {len(centerline_files)}")
    print(f"Ground truth carregado: {len(cobb_gt)} imagens")
    print(f"Output: {OUTPUT_DIR}")

    total_centerlines = len(centerline_files)

    for image_number, centerline_path in enumerate(centerline_files, start=1):
        file_id_raw = centerline_path.stem.replace("_centerline", "")
        file_id_gt = file_id_for_gt(file_id_raw)
        file_id = file_id_raw

        xs, ys = load_centerline_points(centerline_path)

        if len(xs) < 10:
            print(f"{file_id}: centerline insuficiente")
            continue

        xs_smooth, ys_smooth = smooth_centerline(xs, ys)
        xs_trim, ys_trim = trim_centerline(xs_smooth, ys_smooth)

        if len(xs_trim) < 10:
            print(f"{file_id}: pontos insuficientes depois do trim")
            continue

        if len(xs_trim) != len(ys_trim):
            raise ValueError(f"{file_id}: shapes inconsistentes em xs/ys")

        angles = calculate_angles_by_arc_length(xs_trim, ys_trim)

        if len(angles) != len(xs_trim):
            raise ValueError(f"{file_id}: shape inconsistente nos angulos")

        global_extremes = find_cobb_extremes(
            xs_trim,
            ys_trim,
            angles,
            min_point_distance=MIN_POINT_DISTANCE_GLOBAL,
        )

        regional_cobbs = calculate_regional_cobbs(xs_trim, ys_trim, angles)

        gt = cobb_gt.get(file_id_gt)

        row = {
            "file_id": file_id,
            "file_id_gt": file_id_gt,
            "cobb_global_estimated": "",
            "cobb_global_raw": "",
            "error_global_vs_MT": "",
            "abs_error_global_vs_MT": "",
            "within_5_deg_global_vs_MT": "",
            "within_10_deg_global_vs_MT": "",
            "within_15_deg_global_vs_MT": "",
        }

        if global_extremes is not None:
            row["cobb_global_estimated"] = round(global_extremes["cobb"], 2)
            row["cobb_global_raw"] = round(global_extremes["cobb_raw"], 2)

            if gt is not None:
                error_global_mt = global_extremes["cobb"] - gt["MT"]
                abs_error_global_mt = abs(error_global_mt)
                symmetric_smape_value_global_mt = calculate_symmetric_smape_value(
                    global_extremes["cobb"],
                    gt["MT"],
                )
                errors_global_mt.append(error_global_mt)
                symmetric_smape_global_mt.append(symmetric_smape_value_global_mt)

                row["error_global_vs_MT"] = round(error_global_mt, 2)
                row["abs_error_global_vs_MT"] = round(abs_error_global_mt, 2)
                row["within_5_deg_global_vs_MT"] = abs_error_global_mt <= 5
                row["within_10_deg_global_vs_MT"] = abs_error_global_mt <= 10
                row["within_15_deg_global_vs_MT"] = abs_error_global_mt <= 15

        region_errors_for_image = []
        estimated_cobbs_for_image = []
        real_cobbs_for_image = []

        for region in REGIONS:
            region_result = add_region_values(
                row,
                region,
                regional_cobbs[region["name"]],
                gt,
                errors_by_region,
                symmetric_smape_by_region,
            )

            if region_result is not None:
                region_errors_for_image.append(region_result["abs_error"])
                estimated_cobbs_for_image.append(region_result["estimated"])
                real_cobbs_for_image.append(region_result["real"])
                errors_all_regions.append(region_result["error"])
                symmetric_smape_all_regions.append(region_result["symmetric_smape"])

        row["mean_abs_error_3_cobb"] = ""
        row["max_abs_error_3_cobb"] = ""
        row["mean_symmetric_smape_3_cobb"] = ""
        row["benchmark_smape_3_cobb"] = ""
        row["max_cobb_estimated"] = ""
        row["max_cobb_real"] = ""
        row["error_max_cobb"] = ""
        row["abs_error_max_cobb"] = ""
        row["within_5_deg_max_cobb"] = ""
        row["within_10_deg_max_cobb"] = ""
        row["within_15_deg_max_cobb"] = ""

        if len(region_errors_for_image) == len(REGIONS):
            row["mean_abs_error_3_cobb"] = round(float(np.mean(region_errors_for_image)), 2)
            row["max_abs_error_3_cobb"] = round(float(np.max(region_errors_for_image)), 2)
            region_symmetric_smape_for_image = [
                float(row[f"symmetric_smape_{region['name']}"])
                for region in REGIONS
            ]
            row["mean_symmetric_smape_3_cobb"] = round(
                float(np.mean(region_symmetric_smape_for_image)),
                2,
            )
            benchmark_smape_3_cobb = calculate_benchmark_smape_value(
                estimated_cobbs_for_image,
                real_cobbs_for_image,
            )
            row["benchmark_smape_3_cobb"] = round(benchmark_smape_3_cobb, 2)
            benchmark_smape_3_cobb_values.append(benchmark_smape_3_cobb)

        if gt is not None and len(estimated_cobbs_for_image) > 0:
            max_estimated = max(estimated_cobbs_for_image)
            max_real = max(real_cobbs_for_image)
            error_max = max_estimated - max_real
            abs_error_max = abs(error_max)
            symmetric_smape_value_max = calculate_symmetric_smape_value(max_estimated, max_real)
            errors_max_cobb.append(error_max)
            symmetric_smape_max_cobb.append(symmetric_smape_value_max)

            row["max_cobb_estimated"] = round(max_estimated, 2)
            row["max_cobb_real"] = round(max_real, 2)
            row["error_max_cobb"] = round(error_max, 2)
            row["abs_error_max_cobb"] = round(abs_error_max, 2)
            row["within_5_deg_max_cobb"] = abs_error_max <= 5
            row["within_10_deg_max_cobb"] = abs_error_max <= 10
            row["within_15_deg_max_cobb"] = abs_error_max <= 15

        visualization_path = save_visualization(
            file_id,
            xs_smooth,
            ys_smooth,
            xs_trim,
            ys_trim,
            regional_cobbs,
            gt=gt,
        )

        row["visualization_path"] = str(visualization_path)
        results.append(row)

        print_image_result(
            image_number,
            total_centerlines,
            file_id,
            row,
            gt,
        )

    fieldnames = build_fieldnames()
    save_csv(CSV_PATH, results, fieldnames)

    top_outliers = save_top_outliers(results)

    print("\n===== TOP OUTLIERS MULTI-COBB =====")
    for row in top_outliers:
        print(
            f"{row['file_id']}: "
            f"MAE3 = {row['mean_abs_error_3_cobb']} | "
            f"PT err = {row['abs_error_PT']} | "
            f"MT err = {row['abs_error_MT']} | "
            f"TL_L err = {row['abs_error_TL_L']}"
        )

    print("\n===== METRICAS MULTI-COBB =====")

    metric_rows = []

    for region in REGIONS:
        region_name = region["name"]
        errors = errors_by_region[region_name]

        if len(errors) == 0:
            print(f"{region_name}: sem GT valido")
            continue

        mae, rmse = calculate_metrics(errors)
        symmetric_smape = calculate_mean(symmetric_smape_by_region[region_name])
        tolerance_results = calculate_tolerance_accuracy(errors)

        metric_rows.append({
            "target": region_name,
            "n": len(errors),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "symmetric_smape": round(symmetric_smape, 2),
            "benchmark_smape": "",
            "within_5_deg_percent": round(tolerance_results[5], 2),
            "within_10_deg_percent": round(tolerance_results[10], 2),
            "within_15_deg_percent": round(tolerance_results[15], 2),
            "within_5_percent": round(tolerance_results[5], 2),
            "within_10_percent": round(tolerance_results[10], 2),
            "within_15_percent": round(tolerance_results[15], 2),
        })

        print(
            f"{region_name}: N={len(errors)} | "
            f"MAE={mae:.2f} | RMSE={rmse:.2f} | "
            f"SMAPE simetrico={symmetric_smape:.2f}% | "
            f"<=5={tolerance_results[5]:.2f}% | "
            f"<=10={tolerance_results[10]:.2f}% | "
            f"<=15={tolerance_results[15]:.2f}%"
        )

    if len(errors_all_regions) > 0:
        mae_all, rmse_all = calculate_metrics(errors_all_regions)
        symmetric_smape_all = calculate_mean(symmetric_smape_all_regions)
        benchmark_smape_all = calculate_mean(benchmark_smape_3_cobb_values)
        tolerance_all = calculate_tolerance_accuracy(errors_all_regions)

        metric_rows.append({
            "target": "ALL_3_COBB",
            "n": len(errors_all_regions),
            "mae": round(mae_all, 2),
            "rmse": round(rmse_all, 2),
            "symmetric_smape": round(symmetric_smape_all, 2),
            "benchmark_smape": round(benchmark_smape_all, 2),
            "within_5_deg_percent": round(tolerance_all[5], 2),
            "within_10_deg_percent": round(tolerance_all[10], 2),
            "within_15_deg_percent": round(tolerance_all[15], 2),
            "within_5_percent": round(tolerance_all[5], 2),
            "within_10_percent": round(tolerance_all[10], 2),
            "within_15_percent": round(tolerance_all[15], 2),
        })

        print(
            f"ALL_3_COBB: N={len(errors_all_regions)} | "
            f"MAE={mae_all:.2f} | RMSE={rmse_all:.2f} | "
            f"SMAPE benchmark={benchmark_smape_all:.2f}% | "
            f"SMAPE simetrico={symmetric_smape_all:.2f}% | "
            f"<=5={tolerance_all[5]:.2f}% | "
            f"<=10={tolerance_all[10]:.2f}% | "
            f"<=15={tolerance_all[15]:.2f}%"
        )

    if len(errors_max_cobb) > 0:
        mae_max, rmse_max = calculate_metrics(errors_max_cobb)
        symmetric_smape_max = calculate_mean(symmetric_smape_max_cobb)
        tolerance_max = calculate_tolerance_accuracy(errors_max_cobb)

        metric_rows.append({
            "target": "MAX_COBB",
            "n": len(errors_max_cobb),
            "mae": round(mae_max, 2),
            "rmse": round(rmse_max, 2),
            "symmetric_smape": round(symmetric_smape_max, 2),
            "benchmark_smape": "",
            "within_5_deg_percent": round(tolerance_max[5], 2),
            "within_10_deg_percent": round(tolerance_max[10], 2),
            "within_15_deg_percent": round(tolerance_max[15], 2),
            "within_5_percent": round(tolerance_max[5], 2),
            "within_10_percent": round(tolerance_max[10], 2),
            "within_15_percent": round(tolerance_max[15], 2),
        })

        print(
            f"MAX_COBB: N={len(errors_max_cobb)} | "
            f"MAE={mae_max:.2f} | RMSE={rmse_max:.2f} | "
            f"SMAPE simetrico={symmetric_smape_max:.2f}% | "
            f"<=5={tolerance_max[5]:.2f}% | "
            f"<=10={tolerance_max[10]:.2f}% | "
            f"<=15={tolerance_max[15]:.2f}%"
        )

    if len(errors_global_mt) > 0:
        mae_global, rmse_global = calculate_metrics(errors_global_mt)
        symmetric_smape_global = calculate_mean(symmetric_smape_global_mt)
        tolerance_global = calculate_tolerance_accuracy(errors_global_mt)

        metric_rows.append({
            "target": "GLOBAL_VS_MT_BASELINE_COMPARISON",
            "n": len(errors_global_mt),
            "mae": round(mae_global, 2),
            "rmse": round(rmse_global, 2),
            "symmetric_smape": round(symmetric_smape_global, 2),
            "benchmark_smape": "",
            "within_5_deg_percent": round(tolerance_global[5], 2),
            "within_10_deg_percent": round(tolerance_global[10], 2),
            "within_15_deg_percent": round(tolerance_global[15], 2),
            "within_5_percent": round(tolerance_global[5], 2),
            "within_10_percent": round(tolerance_global[10], 2),
            "within_15_percent": round(tolerance_global[15], 2),
        })

        print(
            f"GLOBAL_VS_MT: N={len(errors_global_mt)} | "
            f"MAE={mae_global:.2f} | RMSE={rmse_global:.2f} | "
            f"SMAPE simetrico={symmetric_smape_global:.2f}% | "
            f"<=5={tolerance_global[5]:.2f}%"
        )

    save_metrics_summary(metric_rows)

    print("\n===== CONCLUIDO =====")
    print(f"CSV guardado em: {CSV_PATH}")
    print(f"Metricas guardadas em: {METRICS_CSV_PATH}")
    print(f"Top outliers guardados em: {OUTLIERS_CSV_PATH}")
    print(f"Imagens dos top outliers copiadas para: {OUTLIERS_VISUAL_DIR}")
    print(f"Imagens completas guardadas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
