import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import round_safe, safe_float, safe_int


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
OUTPUT_DIR = (
    PROCESSED_DIR
    / "cobb_results"
    / "comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1"
)

SOURCES = {
    "train2000": "unet_baseline_2000_padding_512_centerline_val_all_3192",
    "trainfull": "unet_train_full_padding_512_centerline_val_all_3192",
}

EXPECTED_CASES = 3192
GOOD_LIMIT = 5.0
WEAK_LIMIT = 10.0
MAX_FILES = safe_int(os.environ.get("SPINAL_247_MAX_FILES", "0"))
OVERWRITE = os.environ.get("SPINAL_247_OVERWRITE", "0").strip() == "1"


PT_PROFILES = [
    {"id": "pt_r25_j5_g001_m01_b8", "max_rank": 25, "max_jump": 5.0, "max_positive_jump": 5.0, "min_score": 0.01, "min_model": 0.10, "min_base_error": 8.0},
    {"id": "pt_r100_j8_g001_m01_b8", "max_rank": 100, "max_jump": 8.0, "max_positive_jump": 5.0, "min_score": 0.01, "min_model": 0.10, "min_base_error": 8.0},
    {"id": "pt_r750_j20_g005_m01_b12", "max_rank": 750, "max_jump": 20.0, "max_positive_jump": 12.0, "min_score": 0.05, "min_model": 0.10, "min_base_error": 12.0},
]

TLL_PROFILES = [
    {"id": "tll_old_r25_j8_s05_b5", "sources": "old", "max_rank": 25, "max_jump": 8.0, "min_score": 0.50, "min_base_error": 5.0},
    {"id": "tll_all_r25_j8_s0_b5", "sources": "all", "max_rank": 25, "max_jump": 8.0, "min_score": 0.0, "min_base_error": 5.0},
    {"id": "tll_new_r500_j16_s0_b8", "sources": "new", "max_rank": 500, "max_jump": 16.0, "min_score": 0.0, "min_base_error": 8.0},
    {"id": "tll_new_r1200_j25_s0_b8", "sources": "new", "max_rank": 1200, "max_jump": 25.0, "min_score": 0.0, "min_base_error": 8.0},
]

SEVERE_FIELDNAMES = [
    "source",
    "centerline_subdir",
    "file_id",
    "raw_file_id",
    "target",
    "candidate_source",
    "candidate_source_group",
    "method",
    "rank",
    "rank_in_combo",
    "export_reason",
    "score",
    "model_score",
    "selection_score",
    "cobb",
    "jump",
    "signed_jump",
    "base_value",
    "base_abs_error_diag",
    "candidate_abs_error_diag",
    "delta_abs_error_diag",
    "gt_value_diag",
    "sf204_profiles_non_gt",
    "sf204_profiles_with_diag_base_gate",
    "sf204_profile_count_non_gt",
    "sf204_profile_count_with_diag_base_gate",
    "model_score_missing_for_pt204",
    "norm_span",
    "norm_y_top",
    "norm_y_bottom",
    "angle_top",
    "angle_bottom",
    "point_distance",
    "smooth",
    "window_start",
    "window_end",
    "band_family",
    "min_point_distance",
    "min_norm_span",
]


def candidate_dir(source_name):
    return OUTPUT_DIR / source_name / "candidates_valid"


def candidate_path(source_name, target_prefix):
    suffix = "" if MAX_FILES <= 0 else f"_smoke_{MAX_FILES}"
    smoke_path = candidate_dir(source_name) / f"{target_prefix}_candidates{suffix}.csv"
    full_path = candidate_dir(source_name) / f"{target_prefix}_candidates.csv"
    if smoke_path.exists():
        return smoke_path
    return full_path


def severe_output_path(source_name):
    suffix = "" if MAX_FILES <= 0 else f"_smoke_{MAX_FILES}"
    return candidate_dir(source_name) / f"severe_candidates{suffix}.csv"


def ensure_output(path):
    if path.exists() and not OVERWRITE:
        raise FileExistsError(f"Output ja existe: {path}. Usa SPINAL_247_OVERWRITE=1 para substituir.")
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    return fields or ["source"]


def iter_limited_rows(path):
    seen_files = []
    seen = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row.get("file_id", "")
            if MAX_FILES > 0 and file_id not in seen:
                if len(seen_files) >= MAX_FILES:
                    break
                seen.add(file_id)
                seen_files.append(file_id)
            yield row


def finite(value):
    return np.isfinite(safe_float(value))


def row_score(row):
    for key in ["score", "selection_score", "predicted_good_score", "model_score"]:
        value = safe_float(row.get(key, ""))
        if np.isfinite(value):
            return value
    return np.nan


def row_model_score(row):
    for key in ["model_score", "predicted_good_score"]:
        value = safe_float(row.get(key, ""))
        if np.isfinite(value):
            return value
    return np.nan


def base_value(row):
    if row.get("target") == "PT":
        return safe_float(row.get("base_pt", ""))
    if row.get("target") == "TL_L":
        return safe_float(row.get("base_tll", ""))
    return np.nan


def gt_value(row):
    if row.get("target") == "PT":
        return safe_float(row.get("gt_pt_diag", ""))
    if row.get("target") == "TL_L":
        return safe_float(row.get("gt_tll_diag", ""))
    return np.nan


def profile_passes_non_gt(row, profile):
    rank = safe_int(row.get("rank", ""))
    jump = safe_float(row.get("jump", ""))
    score = row_score(row)
    signed_jump = safe_float(row.get("signed_jump", ""))
    if rank <= 0 or rank > safe_int(profile["max_rank"]):
        return False
    if not np.isfinite(jump) or jump > safe_float(profile["max_jump"]):
        return False
    if np.isfinite(score) and score < safe_float(profile["min_score"]):
        return False
    if not np.isfinite(score) and safe_float(profile["min_score"]) > 0:
        return False
    if row.get("target") == "PT":
        if np.isfinite(signed_jump) and signed_jump > safe_float(profile["max_positive_jump"]):
            return False
        return True
    if row.get("target") == "TL_L" and profile.get("sources", "all") != "all":
        if row.get("source_group", "") != profile["sources"]:
            return False
    return True


def profile_passes_model_if_available(row, profile):
    if row.get("target") != "PT":
        return True
    model_score = row_model_score(row)
    if not np.isfinite(model_score):
        return True
    return model_score >= safe_float(profile["min_model"])


def profile_passes_diag_base_gate(row, profile):
    value = safe_float(row.get("base_abs_error_diag", ""))
    return np.isfinite(value) and value >= safe_float(profile["min_base_error"])


def matching_profiles(row):
    profiles = PT_PROFILES if row.get("target") == "PT" else TLL_PROFILES
    non_gt = []
    with_diag_base_gate = []
    for profile in profiles:
        if not profile_passes_non_gt(row, profile):
            continue
        if not profile_passes_model_if_available(row, profile):
            continue
        non_gt.append(profile["id"])
        if profile_passes_diag_base_gate(row, profile):
            with_diag_base_gate.append(profile["id"])
    return non_gt, with_diag_base_gate


def normalize_candidate(source_name, centerline_subdir, row, non_gt, with_diag_base_gate):
    target = row.get("target", "")
    base = base_value(row)
    gt = gt_value(row)
    model_missing = int(target == "PT" and not np.isfinite(row_model_score(row)))
    return {
        "source": source_name,
        "centerline_subdir": centerline_subdir,
        "file_id": row.get("file_id", ""),
        "raw_file_id": row.get("raw_file_id", ""),
        "target": target,
        "candidate_source": row.get("source", ""),
        "candidate_source_group": row.get("source_group", ""),
        "method": row.get("method", ""),
        "rank": row.get("rank", ""),
        "rank_in_combo": row.get("rank_in_combo", ""),
        "export_reason": row.get("export_reason", ""),
        "score": round_safe(row_score(row)),
        "model_score": round_safe(row_model_score(row)),
        "selection_score": row.get("selection_score", ""),
        "cobb": row.get("cobb", ""),
        "jump": row.get("jump", ""),
        "signed_jump": row.get("signed_jump", ""),
        "base_value": round_safe(base),
        "base_abs_error_diag": row.get("base_abs_error_diag", ""),
        "candidate_abs_error_diag": row.get("candidate_abs_error_diag", ""),
        "delta_abs_error_diag": row.get("delta_abs_error_diag", ""),
        "gt_value_diag": round_safe(gt),
        "sf204_profiles_non_gt": "+".join(non_gt),
        "sf204_profiles_with_diag_base_gate": "+".join(with_diag_base_gate),
        "sf204_profile_count_non_gt": len(non_gt),
        "sf204_profile_count_with_diag_base_gate": len(with_diag_base_gate),
        "model_score_missing_for_pt204": model_missing,
        "norm_span": row.get("norm_span", ""),
        "norm_y_top": row.get("norm_y_top", ""),
        "norm_y_bottom": row.get("norm_y_bottom", ""),
        "angle_top": row.get("angle_top", ""),
        "angle_bottom": row.get("angle_bottom", ""),
        "point_distance": row.get("point_distance", ""),
        "smooth": row.get("smooth", ""),
        "window_start": row.get("window_start", ""),
        "window_end": row.get("window_end", ""),
        "band_family": row.get("band_family", ""),
        "min_point_distance": row.get("min_point_distance", ""),
        "min_norm_span": row.get("min_norm_span", ""),
    }


def process_candidate_file(source_name, centerline_subdir, path, writer, target, case_stats):
    total_read = 0
    total_exported = 0
    profile_counts = Counter()
    profile_counts_diag = Counter()
    for row in iter_limited_rows(path):
        total_read += 1
        file_id = row.get("file_id", "")
        if row.get("target", "") != target:
            continue
        case = case_stats[file_id]
        case["file_id"] = file_id
        case["raw_file_id"] = row.get("raw_file_id", "")
        case[f"{target}_input_candidates"] += 1
        non_gt, with_diag_base_gate = matching_profiles(row)
        for profile_id in non_gt:
            profile_counts[profile_id] += 1
        for profile_id in with_diag_base_gate:
            profile_counts_diag[profile_id] += 1
        if not non_gt:
            continue
        out_row = normalize_candidate(source_name, centerline_subdir, row, non_gt, with_diag_base_gate)
        writer.writerow(out_row)
        total_exported += 1
        case[f"{target}_severe_candidates"] += 1
        error = safe_float(row.get("candidate_abs_error_diag", ""))
        if np.isfinite(error):
            current = safe_float(case.get(f"{target}_oracle_abs_error_diag", ""))
            if not np.isfinite(current) or error < current:
                case[f"{target}_oracle_abs_error_diag"] = round_safe(error)
                case[f"{target}_oracle_cobb"] = row.get("cobb", "")
                case[f"{target}_oracle_rank"] = row.get("rank", "")
    return {
        "target": target,
        "input_path": str(path),
        "rows_read": total_read,
        "rows_exported": total_exported,
        "profile_counts": profile_counts,
        "profile_counts_diag": profile_counts_diag,
    }


def summarize_cases(source_name, case_stats):
    rows = []
    for file_id in sorted(case_stats):
        row = case_stats[file_id]
        pt_error = safe_float(row.get("PT_oracle_abs_error_diag", ""))
        tll_error = safe_float(row.get("TL_L_oracle_abs_error_diag", ""))
        rows.append({
            "source": source_name,
            "file_id": file_id,
            "raw_file_id": row.get("raw_file_id", ""),
            "pt_input_candidates": row.get("PT_input_candidates", 0),
            "pt_severe_candidates": row.get("PT_severe_candidates", 0),
            "pt_oracle_cobb": row.get("PT_oracle_cobb", ""),
            "pt_oracle_abs_error_diag": row.get("PT_oracle_abs_error_diag", ""),
            "pt_oracle_rank": row.get("PT_oracle_rank", ""),
            "pt_oracle_le5_diag": int(np.isfinite(pt_error) and pt_error <= GOOD_LIMIT),
            "pt_oracle_le10_diag": int(np.isfinite(pt_error) and pt_error <= WEAK_LIMIT),
            "tll_input_candidates": row.get("TL_L_input_candidates", 0),
            "tll_severe_candidates": row.get("TL_L_severe_candidates", 0),
            "tll_oracle_cobb": row.get("TL_L_oracle_cobb", ""),
            "tll_oracle_abs_error_diag": row.get("TL_L_oracle_abs_error_diag", ""),
            "tll_oracle_rank": row.get("TL_L_oracle_rank", ""),
            "tll_oracle_le5_diag": int(np.isfinite(tll_error) and tll_error <= GOOD_LIMIT),
            "tll_oracle_le10_diag": int(np.isfinite(tll_error) and tll_error <= WEAK_LIMIT),
        })
    return rows


def source_summary(source_name, centerline_subdir, severe_path, target_stats, case_rows):
    expected = EXPECTED_CASES if MAX_FILES <= 0 else MAX_FILES
    pt_covered = sum(safe_int(row["pt_severe_candidates"]) > 0 for row in case_rows)
    tll_covered = sum(safe_int(row["tll_severe_candidates"]) > 0 for row in case_rows)
    pt_le5 = sum(safe_int(row["pt_oracle_le5_diag"]) == 1 for row in case_rows)
    tll_le5 = sum(safe_int(row["tll_oracle_le5_diag"]) == 1 for row in case_rows)
    pt_le10 = sum(safe_int(row["pt_oracle_le10_diag"]) == 1 for row in case_rows)
    tll_le10 = sum(safe_int(row["tll_oracle_le10_diag"]) == 1 for row in case_rows)
    return {
        "source": source_name,
        "centerline_subdir": centerline_subdir,
        "severe_candidates_path": str(severe_path),
        "expected_cases": expected,
        "case_count_seen": len(case_rows),
        "pt_covered_images": pt_covered,
        "tll_covered_images": tll_covered,
        "pt_missing_images": expected - pt_covered,
        "tll_missing_images": expected - tll_covered,
        "pt_severe_candidate_count": sum(safe_int(row["pt_severe_candidates"]) for row in case_rows),
        "tll_severe_candidate_count": sum(safe_int(row["tll_severe_candidates"]) for row in case_rows),
        "pt_oracle_le5_count": pt_le5,
        "pt_oracle_le5_pct": round(float(pt_le5 / len(case_rows) * 100.0), 4) if case_rows else 0,
        "pt_oracle_le10_count": pt_le10,
        "pt_oracle_le10_pct": round(float(pt_le10 / len(case_rows) * 100.0), 4) if case_rows else 0,
        "tll_oracle_le5_count": tll_le5,
        "tll_oracle_le5_pct": round(float(tll_le5 / len(case_rows) * 100.0), 4) if case_rows else 0,
        "tll_oracle_le10_count": tll_le10,
        "tll_oracle_le10_pct": round(float(tll_le10 / len(case_rows) * 100.0), 4) if case_rows else 0,
        "pt_rows_read": target_stats["PT"]["rows_read"],
        "tll_rows_read": target_stats["TL_L"]["rows_read"],
        "valid_for_248": int(len(case_rows) == expected and pt_covered == expected and tll_covered == expected),
        "max_files": MAX_FILES,
    }


def profile_summary_rows(source_name, target_stats):
    rows = []
    for target, stats in target_stats.items():
        for profile_id, count in sorted(stats["profile_counts"].items()):
            rows.append({
                "source": source_name,
                "target": target,
                "profile_id": profile_id,
                "count_non_gt": count,
                "count_with_diag_base_gate": stats["profile_counts_diag"].get(profile_id, 0),
            })
    return rows


def run_source(source_name, centerline_subdir):
    severe_path = severe_output_path(source_name)
    ensure_output(severe_path)
    pt_path = candidate_path(source_name, "pt")
    tll_path = candidate_path(source_name, "tll")
    for path in [pt_path, tll_path]:
        if not path.exists():
            raise FileNotFoundError(f"Candidatos regenerados nao encontrados: {path}")

    case_stats = defaultdict(lambda: defaultdict(int))
    target_stats = {}
    with open(severe_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SEVERE_FIELDNAMES)
        writer.writeheader()
        target_stats["PT"] = process_candidate_file(source_name, centerline_subdir, pt_path, writer, "PT", case_stats)
        target_stats["TL_L"] = process_candidate_file(source_name, centerline_subdir, tll_path, writer, "TL_L", case_stats)

    case_rows = summarize_cases(source_name, case_stats)
    write_csv(candidate_dir(source_name) / "severe_case_summary.csv", case_rows)
    return source_summary(source_name, centerline_subdir, severe_path, target_stats, case_rows), profile_summary_rows(source_name, target_stats)


def write_manifest(summaries):
    lines = [
        "# Manifest 247",
        "",
        "## Objetivo",
        "",
        "Regenerar a frente severe-first por CENTERLINE_SUBDIR usando os candidatos PT e TL/L ja regenerados.",
        "",
        "## Fidelidade",
        "",
        "- Usa os perfis PT/TL-L do 204 como anotacao de shortlist.",
        "- Nao cria selector novo.",
        "- Nao cria guard novo.",
        "- Nao escolhe checkpoint final.",
        "- Nao usa GT para exportar candidatos severe-first.",
        "- O gate historico `min_base_error` do 204 depende de erro contra GT; por isso fica apenas como coluna diagnostica.",
        "- O `model_score` PT historico nao existe nos candidatos 244; quando ausente, o script nao o usa para excluir candidatos e assinala `model_score_missing_for_pt204=1`.",
        "",
        "## Configuracao",
        "",
        f"- MAX_FILES: `{MAX_FILES}`.",
        "",
        "## Outputs",
        "",
    ]
    for row in summaries:
        lines.append(f"- {row['source']}: `{row['severe_candidates_path']}`.")
    (OUTPUT_DIR / "247_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(summaries):
    lines = [
        "# Resumo 247",
        "",
        "## Estado",
        "",
    ]
    all_valid = all(safe_int(row["valid_for_248"]) == 1 for row in summaries)
    lines.append("- Severe-first por centerline pronto para alimentar o 248." if all_valid else "- Severe-first ainda nao esta valido para alimentar o 248.")
    lines.extend(["", "## Cobertura", ""])
    for row in summaries:
        lines.append(
            f"- {row['source']}: PT {row['pt_covered_images']}/{row['expected_cases']} "
            f"({row['pt_severe_candidate_count']} candidatos, oracle <=5 {row['pt_oracle_le5_pct']}%, "
            f"<=10 {row['pt_oracle_le10_pct']}%); TL/L {row['tll_covered_images']}/{row['expected_cases']} "
            f"({row['tll_severe_candidate_count']} candidatos, oracle <=5 {row['tll_oracle_le5_pct']}%, "
            f"<=10 {row['tll_oracle_le10_pct']}%)."
        )
    lines.extend([
        "",
        "## Nota metodologica",
        "",
        "- Este script nao desbloqueia o 242 sozinho.",
        "- O proximo passo e o 248, que deve decidir se estes artefactos sao suficientes para reaplicar a pipeline fixa sem usar GT na inferencia.",
    ])
    (OUTPUT_DIR / "247_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    profile_rows = []
    for source_name, centerline_subdir in SOURCES.items():
        summary, rows = run_source(source_name, centerline_subdir)
        summaries.append(summary)
        profile_rows.extend(rows)
        print(
            f"{source_name}: PT {summary['pt_covered_images']}/{summary['expected_cases']} "
            f"TL/L {summary['tll_covered_images']}/{summary['expected_cases']} "
            f"candidatos={summary['pt_severe_candidate_count'] + summary['tll_severe_candidate_count']}",
            flush=True,
        )

    write_csv(OUTPUT_DIR / "247_severe_coverage_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "247_severe_profile_summary.csv", profile_rows)
    write_manifest(summaries)
    write_summary(summaries)
    print(f"Resumo: {OUTPUT_DIR / '247_summary.md'}")


if __name__ == "__main__":
    main()
