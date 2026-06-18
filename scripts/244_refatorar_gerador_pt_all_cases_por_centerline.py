import csv
import importlib.util
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import round_safe, safe_float, safe_int


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
CENTERLINES_DIR = PROCESSED_DIR / "centerlines"
OUTPUT_DIR = (
    PROCESSED_DIR
    / "cobb_results"
    / "comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1"
)

SOURCES = {
    "train2000": "unet_baseline_2000_padding_512_centerline_val_all_3192",
    "trainfull": "unet_train_full_padding_512_centerline_val_all_3192",
}

TARGET = "PT"
GOOD_LIMIT = 5.0
WEAK_LIMIT = 10.0
EXPECTED_CASES = 3192

MAX_FILES = safe_int(os.environ.get("SPINAL_244_MAX_FILES", "0"))
OVERWRITE = os.environ.get("SPINAL_244_OVERWRITE", "0").strip() == "1"

# Valores da frente 207. Estes defaults devem ficar alinhados com o 207.
SMOOTH_VALUES = [
    float(item)
    for item in os.environ.get("SPINAL_244_SMOOTH_VALUES", "0,30,60,120").split(",")
    if item.strip()
]
MIN_POINT_DISTANCES = [
    float(item)
    for item in os.environ.get("SPINAL_244_MIN_POINT_DISTANCES", "15,35").split(",")
    if item.strip()
]
MIN_NORM_SPANS = [
    float(item)
    for item in os.environ.get("SPINAL_244_MIN_NORM_SPANS", "0.04,0.12").split(",")
    if item.strip()
]
MAX_EXPORT_PER_COMBO = safe_int(os.environ.get("SPINAL_244_MAX_EXPORT_PER_COMBO", "250"))
MAX_DIVERSE_PER_COMBO = safe_int(os.environ.get("SPINAL_244_MAX_DIVERSE_PER_COMBO", "250"))
MAX_EXPORT_PER_FILE = safe_int(os.environ.get("SPINAL_244_MAX_EXPORT_PER_FILE", "0"))

PT_WINDOWS = [
    ("pt_wide_000_060", 0.00, 0.60, "band_wider"),
    ("pt_wide_000_070", 0.00, 0.70, "band_wider"),
    ("pt_low_015_060", 0.15, 0.60, "band_lower"),
    ("pt_low_020_070", 0.20, 0.70, "band_lower"),
    ("pt_global_000_095", 0.00, 0.95, "band_smoothing_combo"),
]


def load_module(module_name, script_path):
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script nao encontrado: {script_path}")
    sys.path.insert(0, str(script_path.parent))
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel importar: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_s120_stack(centerline_subdir):
    os.environ["SPINAL_120_CENTERLINE_SUBDIR"] = centerline_subdir
    os.environ["SPINAL_COBB_CENTERLINE_SUBDIR"] = centerline_subdir
    s120 = load_module(
        f"script120_for_244_{centerline_subdir}",
        Path(__file__).resolve().parent / "120_diagnosticar_diversidade_candidatos_pt_checkpoint114_padding_512.py",
    )
    s120.CENTERLINE_SUBDIR = centerline_subdir
    spline_diag, cobb = s120.load_spline_stack()
    return s120, spline_diag, cobb


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = union_fieldnames(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows):
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields or ["file_id"]


def read_gt(cobb):
    return cobb.load_cobb_ground_truth(DATASET_DIR / "Cobb_spinal-AI2024-train_gt.txt")


def raw_file_id(file_id):
    parts = str(file_id).split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else str(file_id)


def candidate_output_path(source_name):
    suffix = "" if MAX_FILES <= 0 else f"_smoke_{MAX_FILES}"
    return OUTPUT_DIR / source_name / "candidates_valid" / f"pt_candidates{suffix}.csv"


def candidates_dir(source_name):
    return candidate_output_path(source_name).parent


def ensure_output_path(path):
    if path.exists() and not OVERWRITE:
        raise FileExistsError(f"Output ja existe: {path}. Usa SPINAL_244_OVERWRITE=1 para substituir.")
    path.parent.mkdir(parents=True, exist_ok=True)


def file_sort_key(path):
    stem = path.stem.replace("_centerline", "")
    try:
        return int(stem.split("_", 1)[0])
    except ValueError:
        return 10**9


def centerline_files(centerline_subdir):
    centerline_dir = CENTERLINES_DIR / centerline_subdir
    if not centerline_dir.exists():
        raise FileNotFoundError(f"Pasta de centerlines nao encontrada: {centerline_dir}")
    files = sorted(centerline_dir.glob("*_centerline.png"), key=file_sort_key)
    if not files:
        raise FileNotFoundError(f"Nenhuma centerline encontrada em: {centerline_dir}")
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    return centerline_dir, files


def direct_pt_base(cobb, xs_raw, ys_raw):
    xs_smooth, ys_smooth = cobb.smooth_centerline(xs_raw, ys_raw)
    xs_trim, ys_trim = cobb.trim_centerline(xs_smooth, ys_smooth)
    if len(xs_trim) < cobb.MIN_REGION_POINTS:
        return np.nan, np.nan, "pontos_insuficientes_depois_trim"
    angles = cobb.calculate_angles_by_arc_length(xs_trim, ys_trim)
    regional = cobb.calculate_regional_cobbs(xs_trim, ys_trim, angles)
    data = regional.get(TARGET, {})
    extremes = data.get("extremes")
    if extremes is None:
        return np.nan, np.nan, data.get("status", "sem_extremos_pt")
    return float(extremes["cobb"]), float(extremes.get("selection_score", np.nan)), data.get("status", "")


def select_export_pairs(candidates, base_pt):
    ranked = list(enumerate(candidates, start=1))
    selected = {}

    def add_many(items, reason):
        for local_rank, pair in items:
            key = (local_rank, round(float(safe_float(pair["cobb"])), 6))
            selected.setdefault(key, (local_rank, pair, reason))

    add_many(ranked[:MAX_EXPORT_PER_COMBO], "top_score")
    if MAX_DIVERSE_PER_COMBO > 0:
        add_many(
            sorted(ranked, key=lambda item: safe_float(item[1]["cobb"]))[:MAX_DIVERSE_PER_COMBO],
            "low_cobb",
        )
        add_many(
            sorted(ranked, key=lambda item: -safe_float(item[1]["cobb"]))[:MAX_DIVERSE_PER_COMBO],
            "high_cobb",
        )
        add_many(
            sorted(ranked, key=lambda item: abs(safe_float(item[1]["cobb"]) - base_pt))[:MAX_DIVERSE_PER_COMBO],
            "low_jump",
        )
    return list(selected.values())


def candidate_row(context, pair, local_rank, smooth, window_name, start, end, family, min_distance, min_span, reason):
    cobb_value = safe_float(pair["cobb"])
    gt_pt = context["gt_pt"]
    base_pt = context["base_pt"]
    candidate_abs_error = abs(cobb_value - gt_pt) if np.isfinite(gt_pt) else np.nan
    base_abs_error = abs(base_pt - gt_pt) if np.isfinite(base_pt) and np.isfinite(gt_pt) else np.nan
    return {
        "file_id": context["file_id"],
        "raw_file_id": context["raw_file_id"],
        "target": TARGET,
        "source": "pt244_all_cases_from_207_logic",
        "source_group": "pt244",
        "method": window_name,
        "rank": 0,
        "rank_in_combo": local_rank,
        "export_reason": reason,
        "score": round_safe(pair.get("selection_score", "")),
        "model_score": "",
        "selection_score": round_safe(pair.get("selection_score", "")),
        "cobb": round_safe(cobb_value),
        "jump": round_safe(abs(cobb_value - base_pt)) if np.isfinite(base_pt) else "",
        "signed_jump": round_safe(cobb_value - base_pt) if np.isfinite(base_pt) else "",
        "base_pt": round_safe(base_pt),
        "base_selection_score_pt": round_safe(context["base_score_pt"]),
        "base_status_pt": context["base_status_pt"],
        "base_abs_error_diag": round_safe(base_abs_error),
        "candidate_abs_error_diag": round_safe(candidate_abs_error),
        "delta_abs_error_diag": round_safe(candidate_abs_error - base_abs_error),
        "gt_pt_diag": round_safe(gt_pt),
        "norm_span": round_safe(pair.get("norm_span", "")),
        "norm_y_top": round_safe(pair.get("norm_y_top", "")),
        "norm_y_bottom": round_safe(pair.get("norm_y_bottom", "")),
        "angle_top": round_safe(pair.get("angle_top", "")),
        "angle_bottom": round_safe(pair.get("angle_bottom", "")),
        "point_distance": round_safe(pair.get("point_distance", "")),
        "smooth": smooth,
        "window_start": start,
        "window_end": end,
        "band_family": family,
        "min_point_distance": min_distance,
        "min_norm_span": min_span,
    }


def dedupe_key(row):
    return (
        row["file_id"],
        row["cobb"],
        row["norm_y_top"],
        row["norm_y_bottom"],
        row["smooth"],
        row["method"],
        row["min_point_distance"],
        row["min_norm_span"],
    )


def generate_for_case(s120, spline_diag, cobb, centerline_path, gt_by_file):
    file_id = centerline_path.stem.replace("_centerline", "")
    raw_id = raw_file_id(file_id)
    gt = gt_by_file.get(raw_id, {})
    gt_pt = safe_float(gt.get("PT", np.nan)) if isinstance(gt, dict) else np.nan
    xs_raw, ys_raw = cobb.load_centerline_points(centerline_path)
    if len(xs_raw) < cobb.MIN_REGION_POINTS:
        raise ValueError(f"Centerline com poucos pontos: {len(xs_raw)}")

    base_pt, base_score_pt, base_status_pt = direct_pt_base(cobb, xs_raw, ys_raw)
    context = {
        "file_id": file_id,
        "raw_file_id": raw_id,
        "gt_pt": gt_pt,
        "base_pt": base_pt,
        "base_score_pt": base_score_pt,
        "base_status_pt": base_status_pt,
    }

    file_rows = []
    combo_summary = []
    for smooth in SMOOTH_VALUES:
        xs, ys = s120.fit_smooth(spline_diag, cobb, xs_raw, ys_raw, smooth)
        angles = cobb.calculate_angles_by_arc_length(xs, ys)
        norm_y = s120.normalized_y(ys)
        for window_name, start, end, family in PT_WINDOWS:
            indices = s120.window_indices(norm_y, start, end)
            window_candidates = s120.collect_candidate_pairs(cobb, ys, angles, indices)
            for min_distance in MIN_POINT_DISTANCES:
                for min_span in MIN_NORM_SPANS:
                    candidates = s120.filter_candidate_pairs(window_candidates, min_distance, min_span)
                    selected = select_export_pairs(candidates, base_pt)
                    combo_rows = [
                        candidate_row(
                            context,
                            pair,
                            local_rank,
                            smooth,
                            window_name,
                            start,
                            end,
                            family,
                            min_distance,
                            min_span,
                            reason,
                        )
                        for local_rank, pair, reason in selected
                    ]
                    file_rows.extend(combo_rows)
                    errors_exported = [safe_float(row["candidate_abs_error_diag"]) for row in combo_rows]
                    combo_summary.append({
                        "file_id": file_id,
                        "raw_file_id": raw_id,
                        "smooth": smooth,
                        "window_name": window_name,
                        "band_family": family,
                        "min_point_distance": min_distance,
                        "min_norm_span": min_span,
                        "candidate_count_total_combo": len(candidates),
                        "candidate_count_exported_combo": len(combo_rows),
                        "oracle_exported_abs_error_diag": round_safe(min(errors_exported) if errors_exported else np.nan),
                        "exported_has_le5_diag": int(any(value <= GOOD_LIMIT for value in errors_exported)),
                        "exported_has_le10_diag": int(any(value <= WEAK_LIMIT for value in errors_exported)),
                    })

    deduped = {}
    for row in file_rows:
        key = dedupe_key(row)
        old = deduped.get(key)
        if old is None or (
            safe_float(row["selection_score"]),
            -safe_int(row["rank_in_combo"]),
        ) > (
            safe_float(old["selection_score"]),
            -safe_int(old["rank_in_combo"]),
        ):
            deduped[key] = row
    ranked = sorted(
        deduped.values(),
        key=lambda row: (
            -safe_float(row["selection_score"]),
            safe_float(row["jump"]) if row["jump"] != "" else np.inf,
            safe_int(row["rank_in_combo"]),
        ),
    )
    if MAX_EXPORT_PER_FILE > 0:
        ranked = ranked[:MAX_EXPORT_PER_FILE]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return context, ranked, combo_summary


def summarize_source(source_name, centerline_subdir, file_count, candidate_rows, case_rows, failures):
    counts = Counter(row["file_id"] for row in candidate_rows)
    per_file = list(counts.values())
    missing = [row["file_id"] for row in case_rows if counts.get(row["file_id"], 0) == 0]
    oracle_le5 = sum(safe_float(row["oracle_pt_abs_error_diag"]) <= GOOD_LIMIT for row in case_rows)
    oracle_le10 = sum(safe_float(row["oracle_pt_abs_error_diag"]) <= WEAK_LIMIT for row in case_rows)
    total_cases = len(case_rows)
    return {
        "source": source_name,
        "centerline_subdir": centerline_subdir,
        "centerline_count_seen": file_count,
        "case_count_processed": total_cases,
        "expected_cases": EXPECTED_CASES if MAX_FILES <= 0 else MAX_FILES,
        "covered_images": sum(1 for value in per_file if value > 0),
        "missing_images": len(missing),
        "candidate_count": len(candidate_rows),
        "min_candidates_per_image": min(per_file) if per_file else 0,
        "mean_candidates_per_image": round(float(np.mean(per_file)), 4) if per_file else 0,
        "max_candidates_per_image": max(per_file) if per_file else 0,
        "oracle_pt_le5_count": oracle_le5,
        "oracle_pt_le5_pct": round(float(oracle_le5 / total_cases * 100.0), 4) if total_cases else 0,
        "oracle_pt_le10_count": oracle_le10,
        "oracle_pt_le10_pct": round(float(oracle_le10 / total_cases * 100.0), 4) if total_cases else 0,
        "failure_count": len(failures),
        "valid_for_243": int(total_cases == (EXPECTED_CASES if MAX_FILES <= 0 else MAX_FILES) and not missing and not failures),
        "max_files": MAX_FILES,
    }


def source_distributions(source_name, candidate_rows):
    rows = []
    for field in ["export_reason", "smooth", "method", "band_family"]:
        counts = Counter(str(row.get(field, "")) for row in candidate_rows)
        for value, count in sorted(counts.items()):
            rows.append({
                "source": source_name,
                "field": field,
                "value": value,
                "count": count,
            })
    return rows


def duplicate_summary(source_name, candidate_rows):
    keys = Counter(dedupe_key(row) for row in candidate_rows)
    duplicates = [count for count in keys.values() if count > 1]
    return {
        "source": source_name,
        "duplicate_keys": len(duplicates),
        "duplicate_rows_extra": int(sum(count - 1 for count in duplicates)),
    }


def run_source(source_name, centerline_subdir):
    out_path = candidate_output_path(source_name)
    ensure_output_path(out_path)
    centerline_dir, files = centerline_files(centerline_subdir)
    s120, spline_diag, cobb = load_s120_stack(centerline_subdir)
    gt_by_file = read_gt(cobb)

    candidate_rows = []
    case_rows = []
    combo_rows = []
    failures = []

    for idx, centerline_path in enumerate(files, start=1):
        try:
            context, rows, combos = generate_for_case(s120, spline_diag, cobb, centerline_path, gt_by_file)
            best = min(rows, key=lambda row: (safe_float(row["candidate_abs_error_diag"]), safe_int(row["rank"])), default=None)
            case_rows.append({
                "source": source_name,
                "file_id": context["file_id"],
                "raw_file_id": context["raw_file_id"],
                "gt_pt_diag": round_safe(context["gt_pt"]),
                "base_pt": round_safe(context["base_pt"]),
                "base_abs_error_diag": round_safe(abs(context["base_pt"] - context["gt_pt"])) if np.isfinite(context["base_pt"]) and np.isfinite(context["gt_pt"]) else "",
                "candidate_count": len(rows),
                "oracle_pt": "" if best is None else best["cobb"],
                "oracle_pt_abs_error_diag": "" if best is None else best["candidate_abs_error_diag"],
                "oracle_pt_rank": "" if best is None else best["rank"],
                "oracle_pt_method": "" if best is None else best["method"],
            })
            candidate_rows.extend(rows)
            combo_rows.extend({"source": source_name, **row} for row in combos)
            if idx % 50 == 0 or idx == len(files):
                print(f"{source_name}: {idx}/{len(files)} casos, candidatos={len(candidate_rows)}", flush=True)
        except Exception as exc:
            file_id = centerline_path.stem.replace("_centerline", "")
            failures.append({
                "source": source_name,
                "file_id": file_id,
                "raw_file_id": raw_file_id(file_id),
                "error": str(exc),
            })
            print(f"{source_name}: {idx}/{len(files)} {file_id} FALHA {exc}", flush=True)

    write_csv(out_path, candidate_rows)
    write_csv(out_path.parent / "pt_case_summary.csv", case_rows)
    write_csv(out_path.parent / "pt_combo_summary.csv", combo_rows)
    write_csv(out_path.parent / "pt_failures.csv", failures, ["source", "file_id", "raw_file_id", "error"])

    summary = summarize_source(source_name, centerline_subdir, len(files), candidate_rows, case_rows, failures)
    distributions = source_distributions(source_name, candidate_rows)
    dupes = duplicate_summary(source_name, candidate_rows)
    return summary, distributions, dupes


def write_manifest(summaries):
    lines = [
        "# Manifest 244",
        "",
        "## Objetivo",
        "",
        "Gerar candidatos PT all-cases por CENTERLINE_SUBDIR usando a logica do 207.",
        "",
        "## Fidelidade",
        "",
        "- Bandas PT iguais ao 207.",
        "- Smooth values iguais ao 207 por default.",
        "- Exportacao diversa: top_score, low_cobb, high_cobb, low_jump.",
        "- GT usado apenas para diagnostico oracle.",
        "- Nao escolhe candidato final.",
        "- Nao cria selector nem guard.",
        "",
        "## Configuracao",
        "",
        f"- SMOOTH_VALUES: `{SMOOTH_VALUES}`.",
        f"- MIN_POINT_DISTANCES: `{MIN_POINT_DISTANCES}`.",
        f"- MIN_NORM_SPANS: `{MIN_NORM_SPANS}`.",
        f"- MAX_EXPORT_PER_COMBO: `{MAX_EXPORT_PER_COMBO}`.",
        f"- MAX_DIVERSE_PER_COMBO: `{MAX_DIVERSE_PER_COMBO}`.",
        f"- MAX_EXPORT_PER_FILE: `{MAX_EXPORT_PER_FILE}`.",
        f"- MAX_FILES: `{MAX_FILES}`.",
        "",
        "## Outputs",
        "",
    ]
    for summary in summaries:
        source = summary["source"]
        lines.append(f"- {source}: `{candidate_output_path(source)}`.")
    (OUTPUT_DIR / "244_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(summaries, duplicate_rows):
    by_source = {row["source"]: row for row in summaries}
    lines = [
        "# Resumo 244",
        "",
        "## Estado",
        "",
    ]
    all_valid = all(safe_int(row["valid_for_243"]) == 1 for row in summaries)
    lines.append("- Gerador PT all-cases valido para alimentar o 243." if all_valid else "- Gerador PT ainda nao e valido para alimentar o 243.")
    lines.extend(["", "## Cobertura", ""])
    for source, row in by_source.items():
        lines.append(
            f"- {source}: {row['covered_images']}/{row['expected_cases']} imagens cobertas, "
            f"{row['candidate_count']} candidatos, oracle <=5 {row['oracle_pt_le5_pct']}%, "
            f"oracle <=10 {row['oracle_pt_le10_pct']}%, falhas {row['failure_count']}."
        )
    lines.extend(["", "## Duplicados", ""])
    for row in duplicate_rows:
        lines.append(f"- {row['source']}: duplicate_keys {row['duplicate_keys']}, duplicate_rows_extra {row['duplicate_rows_extra']}.")
    lines.extend([
        "",
        "## Proximo passo",
        "",
        "- Se ambas as fontes tiverem cobertura completa, avancar para `245_refatorar_gerador_mt_all_cases_por_centerline.py`.",
    ])
    (OUTPUT_DIR / "244_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    distributions = []
    duplicate_rows = []
    for source_name, centerline_subdir in SOURCES.items():
        summary, dist_rows, dupe_row = run_source(source_name, centerline_subdir)
        summaries.append(summary)
        distributions.extend(dist_rows)
        duplicate_rows.append(dupe_row)

    write_csv(OUTPUT_DIR / "244_pt_coverage_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "244_pt_oracle_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "244_pt_distribution_summary.csv", distributions)
    write_csv(OUTPUT_DIR / "244_pt_duplicate_summary.csv", duplicate_rows)
    write_manifest(summaries)
    write_summary(summaries, duplicate_rows)

    for row in summaries:
        print(
            f"{row['source']}: cobertura {row['covered_images']}/{row['expected_cases']} "
            f"candidatos={row['candidate_count']} oracle<=5={row['oracle_pt_le5_pct']}% "
            f"falhas={row['failure_count']}"
        )
    print(f"Resumo: {OUTPUT_DIR / '244_summary.md'}")


if __name__ == "__main__":
    main()
