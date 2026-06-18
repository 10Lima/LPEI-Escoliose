import csv
import itertools
import os
from collections import defaultdict
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

CHECKPOINT191_DIR = Path(os.environ.get(
    "SPINAL_193_CHECKPOINT191_DIR",
    str(ACTIVE_DIR / "checkpoint191_checkpoint186_mt190_v1"),
))
DIAG192_DIR = Path(os.environ.get(
    "SPINAL_193_DIAG192_DIR",
    str(ACTIVE_DIR / "diagnosticar_falhas_checkpoint191_v1"),
))
PT_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_193_PT_CANDIDATES_DIR",
    str(ACTIVE_DIR / "gate_pt_hard_negatives_checkpoint114_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_193_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "validar_pt_focado_checkpoint191_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
TARGET = "PT"
MAX_ROWS_PER_FILE = safe_int(os.environ.get("SPINAL_193_MAX_ROWS_PER_FILE", "750"))


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
    image_rows = read_csv(CHECKPOINT191_DIR / "checkpoint191_image_rows.csv")
    target_rows = read_csv(CHECKPOINT191_DIR / "checkpoint191_target_rows.csv")
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
            "sources191": {target: by_file[file_id][target]["source191"] for target in TARGETS},
        }
    return contexts


def focus_sets():
    rows = read_csv(DIAG192_DIR / "diagnostico192_target_rows.csv")
    image_rows = {row["file_id"]: row for row in read_csv(DIAG192_DIR / "diagnostico192_image_rows.csv")}
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
        and (safe_int(row["target_is_dominant"]) == 1 or safe_int(row["image_severe191"]) == 1)
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
        if safe_int(row["severe191"]) == 1 and "PT" in row["dominant_targets"].split("+")
    }
    return {
        "pt_failed": pt_failed,
        "pt_dominant_failed": pt_dominant_failed,
        "pt_dominant_or_severe": pt_dominant_or_severe,
        "pt_oracle_shortlist": pt_oracle_shortlist,
        "pt_oracle_total": pt_oracle_total,
        "severe_pt_dominant": severe_pt_dominant,
    }


def stream_pt_candidates(contexts, wanted_files):
    path = PT_CANDIDATES_DIR / "gate123_oof_candidate_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV PT nao encontrado: {path}")
    groups = defaultdict(list)
    per_file_counts = defaultdict(int)
    print(f"A ler candidatos PT: {path}", flush=True)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["file_id"]
            if file_id not in wanted_files:
                continue
            if per_file_counts[file_id] >= MAX_ROWS_PER_FILE:
                continue
            per_file_counts[file_id] += 1
            context = contexts[file_id]
            cobb = safe_float(row["cobb"])
            item = {
                "file_id": file_id,
                "rank": safe_int(row["rank"]),
                "cobb": cobb,
                "label_abs_error": safe_float(row.get("label_abs_error", "")),
                "gate123_score": safe_float(row.get("gate123_score", "")),
                "predicted_good_score": safe_float(row.get("predicted_good_score", "")),
                "selection_score": safe_float(row.get("selection_score", "")),
                "jump": abs(cobb - context["base"][0]),
                "signed_jump": cobb - context["base"][0],
                "base_pt_abs_error": context["base_errors"][0],
                "base_mae3": context["base_mae3"],
                "window_name": row.get("window_name", ""),
                "norm_y_top": safe_float(row.get("norm_y_top", "")),
                "norm_y_bottom": safe_float(row.get("norm_y_bottom", "")),
                "norm_span": safe_float(row.get("norm_span", "")),
                "angle_top": safe_float(row.get("angle_top", "")),
                "angle_bottom": safe_float(row.get("angle_bottom", "")),
                "point_distance": safe_float(row.get("point_distance", "")),
            }
            groups[file_id].append(item)
    for file_id in list(groups.keys()):
        groups[file_id].sort(key=lambda item: (safe_int(item["rank"]), -safe_float(item["gate123_score"])))
    return groups


def configs():
    guard_profiles = [
        {"name": "r25_j5_g0_m01_b5_p5_pm0", "max_rank": 25, "max_jump": 5.0, "max_positive_jump": 5.0, "min_gate": 0.0, "min_model": 0.10, "min_base_pt_error": 5.0, "block_pass_margin": 0.0},
        {"name": "r100_j8_g001_m01_b5_p5_pm0", "max_rank": 100, "max_jump": 8.0, "max_positive_jump": 5.0, "min_gate": 0.01, "min_model": 0.10, "min_base_pt_error": 5.0, "block_pass_margin": 0.0},
        {"name": "r100_j8_g001_m03_b8_p5_pm05", "max_rank": 100, "max_jump": 8.0, "max_positive_jump": 5.0, "min_gate": 0.01, "min_model": 0.30, "min_base_pt_error": 8.0, "block_pass_margin": 0.5},
        {"name": "r250_j8_g002_m01_b8_p5_pm05", "max_rank": 250, "max_jump": 8.0, "max_positive_jump": 5.0, "min_gate": 0.02, "min_model": 0.10, "min_base_pt_error": 8.0, "block_pass_margin": 0.5},
        {"name": "r500_j12_g005_m01_b10_p8_pm05", "max_rank": 500, "max_jump": 12.0, "max_positive_jump": 8.0, "min_gate": 0.05, "min_model": 0.10, "min_base_pt_error": 10.0, "block_pass_margin": 0.5},
        {"name": "r500_j12_g001_m03_b10_p8_pm05", "max_rank": 500, "max_jump": 12.0, "max_positive_jump": 8.0, "min_gate": 0.01, "min_model": 0.30, "min_base_pt_error": 10.0, "block_pass_margin": 0.5},
        {"name": "r750_j20_g005_m01_b12_p12_pm05", "max_rank": 750, "max_jump": 20.0, "max_positive_jump": 12.0, "min_gate": 0.05, "min_model": 0.10, "min_base_pt_error": 12.0, "block_pass_margin": 0.5},
    ]
    rows = []
    for focus_name, selection_mode, profile in itertools.product(
        ["pt_dominant_failed", "pt_dominant_or_severe", "pt_oracle_shortlist", "severe_pt_dominant"],
        ["rank", "gate", "model"],
        guard_profiles,
    ):
        rows.append({
            "config_id": (
                f"pt193__{focus_name}__sel_{selection_mode}"
                f"__{profile['name']}"
            ),
            "focus_name": focus_name,
            "selection_mode": selection_mode,
            **{key: value for key, value in profile.items() if key != "name"},
        })
    rows.append({
        "config_id": "pt193__off",
        "focus_name": "pt_failed",
        "selection_mode": "rank",
        "min_base_pt_error": np.inf,
        "max_rank": 0,
        "max_jump": 0.0,
        "max_positive_jump": 0.0,
        "min_gate": np.inf,
        "min_model": np.inf,
        "block_pass_margin": 0.0,
    })
    return rows


def candidate_passes(item, context, config):
    if safe_int(item["rank"]) > safe_int(config["max_rank"]):
        return False
    if safe_float(item["jump"]) > safe_float(config["max_jump"]):
        return False
    if safe_float(item["signed_jump"]) > safe_float(config["max_positive_jump"]):
        return False
    if safe_float(item["gate123_score"]) < safe_float(config["min_gate"]):
        return False
    if safe_float(item["predicted_good_score"]) < safe_float(config["min_model"]):
        return False
    if context["base_errors"][0] < safe_float(config["min_base_pt_error"]):
        return False
    margin = safe_float(config["block_pass_margin"])
    if margin > 0.0 and 5.0 - margin <= context["base_mae3"] <= 5.0:
        return False
    return True


def selection_key(item, mode):
    if mode == "rank":
        return (-safe_int(item["rank"]), safe_float(item["gate123_score"]))
    if mode == "gate":
        return (safe_float(item["gate123_score"]), safe_float(item["predicted_good_score"]), -safe_int(item["rank"]))
    if mode == "model":
        return (safe_float(item["predicted_good_score"]), safe_float(item["gate123_score"]), -safe_int(item["rank"]))
    if mode == "geometry":
        return (safe_float(item["selection_score"]), safe_float(item["gate123_score"]), -safe_int(item["rank"]))
    raise ValueError(f"Modo invalido: {mode}")


def selected_item(items, context, config):
    passed = [item for item in items if candidate_passes(item, context, config)]
    if not passed:
        return None
    return max(passed, key=lambda item: selection_key(item, config["selection_mode"]))


def base_array(contexts):
    return np.asarray([contexts[file_id]["base"] for file_id in sorted(contexts)], dtype=float)


def gt_array(contexts):
    return np.asarray([contexts[file_id]["gt"] for file_id in sorted(contexts)], dtype=float)


def evaluate_config(contexts, groups, focus, config, baseline_preds):
    preds = []
    gts = []
    selected = {}
    focus_files = focus[config["focus_name"]]
    for file_id in sorted(contexts):
        context = contexts[file_id]
        est = list(context["base"])
        if file_id in focus_files:
            item = selected_item(groups.get(file_id, []), context, config)
            if item is not None:
                est[0] = item["cobb"]
                selected[file_id] = item
        preds.append(est)
        gts.append(context["gt"])
    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    summary = metric_summary_from_arrays(preds, gts, baseline_preds=baseline_preds)
    base_mae = np.mean(np.abs(baseline_preds - gts), axis=1)
    new_mae = np.mean(np.abs(preds - gts), axis=1)
    target_delta = np.abs(preds[:, 0] - gts[:, 0]) - np.abs(baseline_preds[:, 0] - gts[:, 0])
    summary.update({
        **config,
        "focus_cases": len(focus_files),
        "selected_cases": len(selected),
        "selected_targets": len(selected),
        "improved_targets": int(np.sum(target_delta < -1e-6)),
        "worsened_targets": int(np.sum(target_delta > 1e-6)),
        "fixed_failures": int(np.sum((base_mae > 5.0) & (new_mae <= 5.0))),
        "new_failures": int(np.sum((base_mae <= 5.0) & (new_mae > 5.0))),
        "fixed_severes": int(np.sum((base_mae > 8.0) & (new_mae <= 8.0))),
        "new_severes": int(np.sum((base_mae <= 8.0) & (new_mae > 8.0))),
        "mean_delta_mae3": round_safe(float(np.mean(new_mae - base_mae))),
    })
    return summary


def sort_safe(summary, baseline):
    return (
        safe_int(summary["regressions_target_gt5"]) == 0,
        safe_int(summary["regressions_image_gt3"]) == 0,
        safe_int(summary["regressions_image_gt1"]) == 0,
        safe_int(summary["new_failures"]) == 0,
        safe_int(summary["new_severes"]) == 0,
        safe_int(summary["severe_gt8"]) <= safe_int(baseline["severe_gt8"]),
        safe_int(summary["failures_gt5"]) <= safe_int(baseline["failures_gt5"]),
        -safe_int(summary["failures_gt5"]),
        -safe_int(summary["severe_gt8"]),
        -safe_float(summary["mae3"]),
    )


def sort_balanced(summary, baseline):
    return (
        safe_int(summary["regressions_target_gt5"]) == 0,
        safe_int(summary["regressions_image_gt3"]) == 0,
        safe_int(summary["new_severes"]) == 0,
        safe_int(summary["severe_gt8"]) <= safe_int(baseline["severe_gt8"]),
        safe_int(summary["failures_gt5"]) < safe_int(baseline["failures_gt5"]),
        -safe_int(summary["regressions_image_gt1"]),
        -safe_int(summary["new_failures"]),
        -safe_int(summary["failures_gt5"]),
        -safe_float(summary["mae3"]),
    )


def build_detail_rows(contexts, groups, focus, config):
    selected_rows = []
    image_rows = []
    target_rows = []
    focus_files = focus[config["focus_name"]]
    for file_id in sorted(contexts):
        context = contexts[file_id]
        est = list(context["base"])
        sources = dict(context["sources191"])
        item = None
        if file_id in focus_files:
            item = selected_item(groups.get(file_id, []), context, config)
            if item is not None:
                est[0] = item["cobb"]
                sources["PT"] = "pt193"
                selected_rows.append({
                    "file_id": file_id,
                    "raw_file_id": context["raw_file_id"],
                    "target": "PT",
                    "gt": round_safe(context["gt"][0]),
                    "base_pt": round_safe(context["base"][0]),
                    "selected_pt": round_safe(item["cobb"]),
                    "base_pt_abs_error": round_safe(context["base_errors"][0]),
                    "selected_pt_abs_error": round_safe(abs(item["cobb"] - context["gt"][0])),
                    "delta_pt_abs_error": round_safe(abs(item["cobb"] - context["gt"][0]) - context["base_errors"][0]),
                    "base_mae3": round_safe(context["base_mae3"]),
                    "rank": item["rank"],
                    "gate123_score": round_safe(item["gate123_score"]),
                    "predicted_good_score": round_safe(item["predicted_good_score"]),
                    "selection_score": round_safe(item["selection_score"]),
                    "jump": round_safe(item["jump"]),
                    "signed_jump": round_safe(item["signed_jump"]),
                    "window_name": item["window_name"],
                    "norm_span": round_safe(item["norm_span"]),
                    "norm_y_top": round_safe(item["norm_y_top"]),
                    "norm_y_bottom": round_safe(item["norm_y_bottom"]),
                    "config_id": config["config_id"],
                })
        errors = np.abs(np.asarray(est) - np.asarray(context["gt"]))
        image_rows.append({
            "file_id": file_id,
            "raw_file_id": context["raw_file_id"],
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(np.mean(errors)),
            "within_5": int(np.mean(errors) <= 5.0),
            "pt_source193": sources["PT"],
            "mt_source193": sources["MT"],
            "tll_source193": sources["TL_L"],
            "changed_targets193": "PT" if item is not None else "",
        })
        for idx, target in enumerate(TARGETS):
            target_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "gt": round_safe(context["gt"][idx]),
                "estimated": round_safe(est[idx]),
                "abs_error": round_safe(errors[idx]),
                "source193": sources[target],
            })
    return selected_rows, image_rows, target_rows


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
        "# Validacao 193 PT focado checkpoint191",
        "",
        "## Escopo",
        "",
        "- PT apenas; MT e TL/L congelados no checkpoint191.",
        "- Foco baseado no diagnostico192.",
        "- Selecao usa rank, score, jump e features do candidato.",
        "- GT/oracle usado apenas para diagnostico/foco e metricas.",
        "",
        "## Baseline checkpoint191",
        "",
    ]
    for key, value in compact({"config_id": "checkpoint191", **baseline}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor safe 193", ""])
    for key, value in compact(best_safe).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor balanced 193", ""])
    for key, value in compact(best_balanced).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Leitura", ""])
    if safe_int(best_safe["failures_gt5"]) < safe_int(baseline["failures_gt5"]):
        lines.append("- Existe melhoria segura face ao checkpoint191.")
    else:
        lines.append("- Nenhuma configuracao safe melhorou o checkpoint191.")
    lines.append("- Este script valida PT focado; nao promove checkpoint.")
    (OUTPUT_DIR / "pt193_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    contexts = load_contexts()
    focus = focus_sets()
    wanted_files = set().union(*focus.values())
    groups = stream_pt_candidates(contexts, wanted_files)
    baseline_preds = base_array(contexts)
    gts = gt_array(contexts)
    baseline = metric_summary_from_arrays(baseline_preds, gts)

    rows = []
    best_safe = None
    best_balanced = None
    best_safe_config = None
    best_balanced_config = None
    for config in configs():
        summary = evaluate_config(contexts, groups, focus, config, baseline_preds)
        rows.append(summary)
        safe_key = sort_safe(summary, baseline)
        balanced_key = sort_balanced(summary, baseline)
        if best_safe is None or safe_key > best_safe[0]:
            best_safe = (safe_key, summary)
            best_safe_config = config
        if best_balanced is None or balanced_key > best_balanced[0]:
            best_balanced = (balanced_key, summary)
            best_balanced_config = config

    best_safe_summary = best_safe[1]
    best_balanced_summary = best_balanced[1]
    selected_rows, image_rows, target_rows = build_detail_rows(contexts, groups, focus, best_safe_config)
    balanced_selected, balanced_images, balanced_targets = build_detail_rows(contexts, groups, focus, best_balanced_config)

    write_csv(OUTPUT_DIR / "pt193_config_summary.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "pt193_best_safe_summary.csv", [best_safe_summary], union_fieldnames([best_safe_summary]))
    write_csv(OUTPUT_DIR / "pt193_best_balanced_summary.csv", [best_balanced_summary], union_fieldnames([best_balanced_summary]))
    write_csv(OUTPUT_DIR / "pt193_best_safe_selected_cases.csv", selected_rows, union_fieldnames(selected_rows) if selected_rows else ["file_id"])
    write_csv(OUTPUT_DIR / "pt193_best_safe_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "pt193_best_safe_target_rows.csv", target_rows, union_fieldnames(target_rows))
    write_csv(OUTPUT_DIR / "pt193_best_balanced_selected_cases.csv", balanced_selected, union_fieldnames(balanced_selected) if balanced_selected else ["file_id"])
    write_csv(OUTPUT_DIR / "pt193_best_balanced_image_rows.csv", balanced_images, union_fieldnames(balanced_images))
    write_csv(OUTPUT_DIR / "pt193_best_balanced_target_rows.csv", balanced_targets, union_fieldnames(balanced_targets))
    write_summary(baseline, best_safe_summary, best_balanced_summary)

    print("\n===== PT193 FOCADO CHECKPOINT191 =====")
    print_required_metric_block("checkpoint191", baseline)
    print_required_metric_block("melhor safe 193", best_safe_summary)
    print_required_metric_block("melhor balanced 193", best_balanced_summary)
    print(f"\nResumo: {OUTPUT_DIR / 'pt193_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
