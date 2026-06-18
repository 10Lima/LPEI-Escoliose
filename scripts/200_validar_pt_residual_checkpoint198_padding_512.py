import csv
import importlib.util
import os
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import (
    TARGETS,
    metric_summary_from_arrays,
    print_required_metric_block,
    round_safe,
    safe_float,
    safe_int,
)


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT198_DIR = Path(os.environ.get(
    "SPINAL_200_CHECKPOINT198_DIR",
    str(ACTIVE_DIR / "checkpoint198_checkpoint194_tll197_v1"),
))
DIAG199_DIR = Path(os.environ.get(
    "SPINAL_200_DIAG199_DIR",
    str(ACTIVE_DIR / "diagnosticar_falhas_checkpoint198_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_200_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "validar_pt_residual_checkpoint198_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
PT193_PATH = DATASET_DIR / "Scripts" / "193_validar_pt_focado_checkpoint191_padding_512.py"


def load_pt193_module():
    spec = importlib.util.spec_from_file_location("pt193_module", PT193_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar modulo: {PT193_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows):
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def output_dir_has_results(output_dir):
    if not output_dir.exists():
        return False
    for item in output_dir.iterdir():
        if item.is_file():
            return True
        if item.is_dir() and any(item.iterdir()):
            return True
    return False


def prepare_output_dir():
    if output_dir_has_results(OUTPUT_DIR):
        raise FileExistsError(f"Pasta de output ja tem ficheiros: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def target_index(rows):
    indexed = {}
    for row in rows:
        indexed.setdefault(row["file_id"], {})[row["target"]] = row
    return indexed


def load_contexts():
    image_rows = read_csv(CHECKPOINT198_DIR / "checkpoint198_image_rows.csv")
    target_rows = read_csv(CHECKPOINT198_DIR / "checkpoint198_target_rows.csv")
    by_file = target_index(target_rows)
    contexts = {}
    for image in image_rows:
        file_id = image["file_id"]
        gt = [safe_float(by_file[file_id][target]["gt"]) for target in TARGETS]
        estimated = [safe_float(by_file[file_id][target]["estimated"]) for target in TARGETS]
        errors = [abs(estimated[idx] - gt[idx]) for idx in range(3)]
        contexts[file_id] = {
            "file_id": file_id,
            "raw_file_id": image.get("raw_file_id", ""),
            "gt": gt,
            "base": estimated,
            "base_errors": errors,
            "base_mae3": float(np.mean(errors)),
            "sources191": {target: by_file[file_id][target]["source198"] for target in TARGETS},
        }
    return contexts


def focus_sets():
    rows = read_csv(DIAG199_DIR / "diagnostico199_target_rows.csv")
    image_rows = {row["file_id"]: row for row in read_csv(DIAG199_DIR / "diagnostico199_image_rows.csv")}
    pt_failed = {
        row["file_id"] for row in rows
        if row["target"] == "PT" and safe_int(row["target_failed_gt5"]) == 1
    }
    pt_dominant_failed = {
        row["file_id"] for row in rows
        if row["target"] == "PT"
        and safe_int(row["target_failed_gt5"]) == 1
        and safe_int(row["target_is_dominant"]) == 1
    }
    pt_dominant_or_severe = {
        row["file_id"] for row in rows
        if row["target"] == "PT"
        and safe_int(row["target_failed_gt5"]) == 1
        and (safe_int(row["target_is_dominant"]) == 1 or safe_int(row["image_severe194"]) == 1)
    }
    pt_oracle_shortlist = {
        row["file_id"] for row in rows
        if row["target"] == "PT"
        and safe_int(row["target_failed_gt5"]) == 1
        and safe_int(row["good_candidate_shortlist_le5"]) == 1
    }
    pt_oracle_total = {
        row["file_id"] for row in rows
        if row["target"] == "PT"
        and safe_int(row["target_failed_gt5"]) == 1
        and safe_int(row["good_candidate_total_le5"]) == 1
    }
    severe_pt_dominant = {
        file_id for file_id, row in image_rows.items()
        if safe_int(row["severe194"]) == 1 and "PT" in row["dominant_targets"].split("+")
    }
    return {
        "pt_failed": pt_failed,
        "pt_dominant_failed": pt_dominant_failed,
        "pt_dominant_or_severe": pt_dominant_or_severe,
        "pt_oracle_shortlist": pt_oracle_shortlist,
        "pt_oracle_total": pt_oracle_total,
        "severe_pt_dominant": severe_pt_dominant,
    }


def base_array(contexts):
    return np.asarray([contexts[file_id]["base"] for file_id in sorted(contexts)], dtype=float)


def gt_array(contexts):
    return np.asarray([contexts[file_id]["gt"] for file_id in sorted(contexts)], dtype=float)


def compact(row):
    keys = [
        "config_id", "mae3", "rmse3", "within_5", "within_10", "within_15",
        "failures_gt5", "severe_gt8", "regressions_target_gt5",
        "regressions_image_gt1", "regressions_image_gt3", "new_failures",
        "new_severes", "fixed_failures", "fixed_severes", "selected_cases",
        "max_cobb_smape_article", "pt_smape_article", "mt_smape_article",
        "tll_smape_article", "agg3_smape_article",
    ]
    return {key: row.get(key, "") for key in keys}


def write_summary(baseline, best_safe, best_balanced):
    lines = [
        "# Validacao 200 PT residual checkpoint198",
        "",
        "## Escopo",
        "",
        "- PT apenas; MT e TL/L congelados no checkpoint198.",
        "- Foco baseado no diagnostico199.",
        "- Reutiliza a grelha conservadora do PT193 sobre a nova base.",
        "",
        "## Baseline checkpoint198",
        "",
    ]
    for key, value in compact({"config_id": "checkpoint198", **baseline}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor safe 200", ""])
    for key, value in compact(best_safe).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor balanced 200", ""])
    for key, value in compact(best_balanced).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Leitura", ""])
    if safe_int(best_safe["failures_gt5"]) < safe_int(baseline["failures_gt5"]):
        lines.append("- Existe melhoria segura face ao checkpoint198.")
    else:
        lines.append("- Nenhuma configuracao safe reduziu falhas face ao checkpoint198.")
    if safe_int(best_safe["severe_gt8"]) < safe_int(baseline["severe_gt8"]):
        lines.append("- A melhor configuracao safe tambem reduz severos.")
    lines.append("- Este script valida PT residual; nao promove checkpoint.")
    (OUTPUT_DIR / "pt200_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    pt193 = load_pt193_module()
    contexts = load_contexts()
    focus = focus_sets()
    wanted_files = set().union(*focus.values())
    groups = pt193.stream_pt_candidates(contexts, wanted_files)
    baseline_preds = base_array(contexts)
    gts = gt_array(contexts)
    baseline = metric_summary_from_arrays(baseline_preds, gts)

    rows = []
    best_safe = None
    best_balanced = None
    best_safe_config = None
    best_balanced_config = None
    for config in pt193.configs():
        summary = pt193.evaluate_config(contexts, groups, focus, config, baseline_preds)
        rows.append(summary)
        safe_key = pt193.sort_safe(summary, baseline)
        balanced_key = pt193.sort_balanced(summary, baseline)
        if best_safe is None or safe_key > best_safe[0]:
            best_safe = (safe_key, summary)
            best_safe_config = config
        if best_balanced is None or balanced_key > best_balanced[0]:
            best_balanced = (balanced_key, summary)
            best_balanced_config = config

    best_safe_summary = best_safe[1]
    best_balanced_summary = best_balanced[1]
    selected_rows, image_rows, target_rows = pt193.build_detail_rows(contexts, groups, focus, best_safe_config)
    balanced_selected, balanced_images, balanced_targets = pt193.build_detail_rows(contexts, groups, focus, best_balanced_config)

    write_csv(OUTPUT_DIR / "pt200_config_summary.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "pt200_best_safe_summary.csv", [best_safe_summary], union_fieldnames([best_safe_summary]))
    write_csv(OUTPUT_DIR / "pt200_best_balanced_summary.csv", [best_balanced_summary], union_fieldnames([best_balanced_summary]))
    write_csv(OUTPUT_DIR / "pt200_best_safe_selected_cases.csv", selected_rows, union_fieldnames(selected_rows) if selected_rows else ["file_id"])
    write_csv(OUTPUT_DIR / "pt200_best_safe_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "pt200_best_safe_target_rows.csv", target_rows, union_fieldnames(target_rows))
    write_csv(OUTPUT_DIR / "pt200_best_balanced_selected_cases.csv", balanced_selected, union_fieldnames(balanced_selected) if balanced_selected else ["file_id"])
    write_csv(OUTPUT_DIR / "pt200_best_balanced_image_rows.csv", balanced_images, union_fieldnames(balanced_images))
    write_csv(OUTPUT_DIR / "pt200_best_balanced_target_rows.csv", balanced_targets, union_fieldnames(balanced_targets))
    write_summary(baseline, best_safe_summary, best_balanced_summary)

    print("\n===== PT200 RESIDUAL CHECKPOINT198 =====")
    print_required_metric_block("checkpoint198", baseline)
    print_required_metric_block("melhor safe 200", best_safe_summary)
    print_required_metric_block("melhor balanced 200", best_balanced_summary)
    print(f"\nResumo: {OUTPUT_DIR / 'pt200_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
