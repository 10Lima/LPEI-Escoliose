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

CHECKPOINT194_DIR = Path(os.environ.get(
    "SPINAL_196_CHECKPOINT194_DIR",
    str(ACTIVE_DIR / "checkpoint194_checkpoint191_pt193_v1"),
))
DIAG195_DIR = Path(os.environ.get(
    "SPINAL_196_DIAG195_DIR",
    str(ACTIVE_DIR / "diagnosticar_falhas_checkpoint194_v1"),
))
TLL_OLD_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_196_TLL_OLD_CANDIDATES_DIR",
    str(ACTIVE_DIR / "gate_two_stage_hard_negatives_tll_v1"),
))
TLL159_DIR = Path(os.environ.get(
    "SPINAL_196_TLL159_DIR",
    str(ACTIVE_DIR / "gerar_candidatos_tll_checkpoint156_relevantes_v1"),
))
TLL175_DIR = Path(os.environ.get(
    "SPINAL_196_TLL175_DIR",
    str(ACTIVE_DIR / "gerar_candidatos_tll_remaining_selector173_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_196_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "validar_tll_focado_checkpoint194_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
TARGET = "TL_L"
TARGET_IDX = TARGETS.index(TARGET)
MAX_ROWS_PER_FILE_PER_SOURCE = safe_int(os.environ.get("SPINAL_196_MAX_ROWS_PER_FILE_PER_SOURCE", "1200"))


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
    image_rows = read_csv(CHECKPOINT194_DIR / "checkpoint194_image_rows.csv")
    target_rows = read_csv(CHECKPOINT194_DIR / "checkpoint194_target_rows.csv")
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
            "sources194": {target: by_file[file_id][target]["source194"] for target in TARGETS},
        }
    return contexts


def focus_sets():
    rows = read_csv(DIAG195_DIR / "diagnostico195_target_rows.csv")
    image_rows = {row["file_id"]: row for row in read_csv(DIAG195_DIR / "diagnostico195_image_rows.csv")}
    tll_failed = {
        row["file_id"] for row in rows
        if row["target"] == TARGET and safe_int(row["target_failed_gt5"]) == 1
    }
    tll_dominant_failed = {
        row["file_id"] for row in rows
        if row["target"] == TARGET
        and safe_int(row["target_failed_gt5"]) == 1
        and safe_int(row["target_is_dominant"]) == 1
    }
    tll_dominant_or_severe = {
        row["file_id"] for row in rows
        if row["target"] == TARGET
        and safe_int(row["target_failed_gt5"]) == 1
        and (safe_int(row["target_is_dominant"]) == 1 or safe_int(row["image_severe194"]) == 1)
    }
    tll_oracle_shortlist = {
        row["file_id"] for row in rows
        if row["target"] == TARGET
        and safe_int(row["target_failed_gt5"]) == 1
        and safe_int(row["good_candidate_shortlist_le5"]) == 1
    }
    severe_tll_dominant = {
        file_id for file_id, row in image_rows.items()
        if safe_int(row["severe194"]) == 1 and TARGET in row["dominant_targets"].split("+")
    }
    severe_tll_failed = {
        row["file_id"] for row in rows
        if row["target"] == TARGET
        and safe_int(row["target_failed_gt5"]) == 1
        and safe_int(row["image_severe194"]) == 1
    }
    return {
        "tll_failed": tll_failed,
        "tll_dominant_failed": tll_dominant_failed,
        "tll_dominant_or_severe": tll_dominant_or_severe,
        "tll_oracle_shortlist": tll_oracle_shortlist,
        "severe_tll_dominant": severe_tll_dominant,
        "severe_tll_failed": severe_tll_failed,
    }


def first_score(row, cols):
    for col in cols:
        value = safe_float(row.get(col, ""))
        if np.isfinite(value):
            return value
    return np.nan


def candidate_specs():
    return [
        {
            "path": TLL_OLD_CANDIDATES_DIR / "gate114_oof_candidate_predictions.csv",
            "source": "old_TLL",
            "source_group": "old",
            "rank_col": "rank",
            "score_cols": ["gate114_score", "predicted_good_score", "selection_score"],
            "method_col": "",
        },
        {
            "path": TLL159_DIR / "tll159_candidate_rows.csv",
            "source": "tll159",
            "source_group": "new",
            "rank_col": "rank159",
            "score_cols": ["selection_score"],
            "method_col": "method159",
        },
        {
            "path": TLL175_DIR / "tll175_candidate_rows.csv",
            "source": "tll175",
            "source_group": "new",
            "rank_col": "rank175",
            "score_cols": ["selection_score"],
            "method_col": "method175",
        },
    ]


def stream_tll_candidates(contexts, wanted_files):
    groups = defaultdict(list)
    for spec in candidate_specs():
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"CSV TL/L nao encontrado: {path}")
        per_file_counts = defaultdict(int)
        print(f"A ler candidatos {spec['source']}: {path}", flush=True)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row.get("file_id", "")
                if file_id not in wanted_files:
                    continue
                if per_file_counts[file_id] >= MAX_ROWS_PER_FILE_PER_SOURCE:
                    continue
                cobb = safe_float(row.get("cobb", ""))
                if not np.isfinite(cobb):
                    continue
                per_file_counts[file_id] += 1
                context = contexts[file_id]
                groups[file_id].append({
                    "file_id": file_id,
                    "source": spec["source"],
                    "source_group": spec["source_group"],
                    "method": row.get(spec["method_col"], "") if spec["method_col"] else spec["source"],
                    "rank": safe_int(row.get(spec["rank_col"], "")),
                    "score": first_score(row, spec["score_cols"]),
                    "cobb": cobb,
                    "label_abs_error": safe_float(row.get("label_abs_error", "")),
                    "jump": abs(cobb - context["base"][TARGET_IDX]),
                    "signed_jump": cobb - context["base"][TARGET_IDX],
                    "base_tll_abs_error": context["base_errors"][TARGET_IDX],
                    "base_mae3": context["base_mae3"],
                    "norm_y_top": safe_float(row.get("norm_y_top", "")),
                    "norm_y_bottom": safe_float(row.get("norm_y_bottom", "")),
                    "norm_span": safe_float(row.get("norm_span", "")),
                    "angle_top": safe_float(row.get("angle_top", "")),
                    "angle_bottom": safe_float(row.get("angle_bottom", "")),
                })
    for file_id in list(groups.keys()):
        groups[file_id].sort(key=lambda item: (safe_int(item["rank"]), -safe_float(item["score"])))
    return groups


def configs():
    guard_profiles = [
        {"name": "old_r25_j8_s05_b5_pm0", "sources": "old", "max_rank": 25, "max_jump": 8.0, "min_score": 0.50, "min_base_tll_error": 5.0, "block_pass_margin": 0.0},
        {"name": "all_r25_j8_s0_b5_pm0", "sources": "all", "max_rank": 25, "max_jump": 8.0, "min_score": 0.0, "min_base_tll_error": 5.0, "block_pass_margin": 0.0},
        {"name": "all_r100_j12_s0_b8_pm05", "sources": "all", "max_rank": 100, "max_jump": 12.0, "min_score": 0.0, "min_base_tll_error": 8.0, "block_pass_margin": 0.5},
        {"name": "all_r500_j16_s0_b10_pm05", "sources": "all", "max_rank": 500, "max_jump": 16.0, "min_score": 0.0, "min_base_tll_error": 10.0, "block_pass_margin": 0.5},
        {"name": "new_r500_j16_s0_b10_pm05", "sources": "new", "max_rank": 500, "max_jump": 16.0, "min_score": 0.0, "min_base_tll_error": 10.0, "block_pass_margin": 0.5},
        {"name": "new_r1200_j25_s0_b12_pm05", "sources": "new", "max_rank": 1200, "max_jump": 25.0, "min_score": 0.0, "min_base_tll_error": 12.0, "block_pass_margin": 0.5},
    ]
    rows = []
    for focus_name, selection_mode, profile in itertools.product(
        [
            "tll_dominant_failed",
            "tll_dominant_or_severe",
            "tll_oracle_shortlist",
            "severe_tll_dominant",
            "severe_tll_failed",
        ],
        ["rank", "score"],
        guard_profiles,
    ):
        rows.append({
            "config_id": f"tll196__{focus_name}__sel_{selection_mode}__{profile['name']}",
            "focus_name": focus_name,
            "selection_mode": selection_mode,
            **profile,
        })
    rows.append({
        "config_id": "tll196__off",
        "focus_name": "tll_failed",
        "selection_mode": "rank",
        "sources": "all",
        "max_rank": 0,
        "max_jump": 0.0,
        "min_score": np.inf,
        "min_base_tll_error": np.inf,
        "block_pass_margin": 0.0,
    })
    return rows


def candidate_passes(item, context, config):
    if safe_int(item["rank"]) > safe_int(config["max_rank"]):
        return False
    if safe_float(item["jump"]) > safe_float(config["max_jump"]):
        return False
    if safe_float(item["score"]) < safe_float(config["min_score"]):
        return False
    if config["sources"] != "all" and item["source_group"] != config["sources"]:
        return False
    if context["base_errors"][TARGET_IDX] < safe_float(config["min_base_tll_error"]):
        return False
    margin = safe_float(config["block_pass_margin"])
    if margin > 0.0 and 5.0 - margin <= context["base_mae3"] <= 5.0:
        return False
    return True


def selection_key(item, mode):
    if mode == "rank":
        return (-safe_int(item["rank"]), safe_float(item["score"]))
    if mode == "score":
        return (safe_float(item["score"]), -safe_int(item["rank"]))
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
                est[TARGET_IDX] = item["cobb"]
                selected[file_id] = item
        preds.append(est)
        gts.append(context["gt"])
    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    summary = metric_summary_from_arrays(preds, gts, baseline_preds=baseline_preds)
    base_mae = np.mean(np.abs(baseline_preds - gts), axis=1)
    new_mae = np.mean(np.abs(preds - gts), axis=1)
    target_delta = np.abs(preds[:, TARGET_IDX] - gts[:, TARGET_IDX]) - np.abs(baseline_preds[:, TARGET_IDX] - gts[:, TARGET_IDX])
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
        sources = dict(context["sources194"])
        item = None
        if file_id in focus_files:
            item = selected_item(groups.get(file_id, []), context, config)
            if item is not None:
                est[TARGET_IDX] = item["cobb"]
                sources[TARGET] = f"tll196_{item['source']}"
                selected_rows.append({
                    "file_id": file_id,
                    "raw_file_id": context["raw_file_id"],
                    "target": TARGET,
                    "gt": round_safe(context["gt"][TARGET_IDX]),
                    "base_tll": round_safe(context["base"][TARGET_IDX]),
                    "selected_tll": round_safe(item["cobb"]),
                    "base_tll_abs_error": round_safe(context["base_errors"][TARGET_IDX]),
                    "selected_tll_abs_error": round_safe(abs(item["cobb"] - context["gt"][TARGET_IDX])),
                    "delta_tll_abs_error": round_safe(abs(item["cobb"] - context["gt"][TARGET_IDX]) - context["base_errors"][TARGET_IDX]),
                    "base_mae3": round_safe(context["base_mae3"]),
                    "rank": item["rank"],
                    "source": item["source"],
                    "method": item["method"],
                    "score": round_safe(item["score"]),
                    "jump": round_safe(item["jump"]),
                    "signed_jump": round_safe(item["signed_jump"]),
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
            "pt_source196": sources["PT"],
            "mt_source196": sources["MT"],
            "tll_source196": sources["TL_L"],
            "changed_targets196": TARGET if item is not None else "",
        })
        for idx, target in enumerate(TARGETS):
            target_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "gt": round_safe(context["gt"][idx]),
                "estimated": round_safe(est[idx]),
                "abs_error": round_safe(errors[idx]),
                "source196": sources[target],
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
        "# Validacao 196 TL/L focado checkpoint194",
        "",
        "## Escopo",
        "",
        "- TL/L apenas; PT e MT congelados no checkpoint194.",
        "- Foco baseado no diagnostico195.",
        "- Candidatos: gate114 antigo, tll159 e tll175.",
        "- GT/oracle usado apenas para diagnostico/foco e metricas.",
        "",
        "## Baseline checkpoint194",
        "",
    ]
    for key, value in compact({"config_id": "checkpoint194", **baseline}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor safe 196", ""])
    for key, value in compact(best_safe).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor balanced 196", ""])
    for key, value in compact(best_balanced).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Leitura", ""])
    if safe_int(best_safe["failures_gt5"]) < safe_int(baseline["failures_gt5"]):
        lines.append("- Existe melhoria segura face ao checkpoint194.")
    else:
        lines.append("- Nenhuma configuracao safe reduziu falhas face ao checkpoint194.")
    lines.append("- Este script valida TL/L focado; nao promove checkpoint.")
    (OUTPUT_DIR / "tll196_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    contexts = load_contexts()
    focus = focus_sets()
    wanted_files = set().union(*focus.values())
    groups = stream_tll_candidates(contexts, wanted_files)
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

    write_csv(OUTPUT_DIR / "tll196_config_summary.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "tll196_best_safe_summary.csv", [best_safe_summary], union_fieldnames([best_safe_summary]))
    write_csv(OUTPUT_DIR / "tll196_best_balanced_summary.csv", [best_balanced_summary], union_fieldnames([best_balanced_summary]))
    write_csv(OUTPUT_DIR / "tll196_best_safe_selected_cases.csv", selected_rows, union_fieldnames(selected_rows) if selected_rows else ["file_id"])
    write_csv(OUTPUT_DIR / "tll196_best_safe_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "tll196_best_safe_target_rows.csv", target_rows, union_fieldnames(target_rows))
    write_csv(OUTPUT_DIR / "tll196_best_balanced_selected_cases.csv", balanced_selected, union_fieldnames(balanced_selected) if balanced_selected else ["file_id"])
    write_csv(OUTPUT_DIR / "tll196_best_balanced_image_rows.csv", balanced_images, union_fieldnames(balanced_images))
    write_csv(OUTPUT_DIR / "tll196_best_balanced_target_rows.csv", balanced_targets, union_fieldnames(balanced_targets))
    write_summary(baseline, best_safe_summary, best_balanced_summary)

    print("\n===== TLL196 FOCADO CHECKPOINT194 =====")
    print_required_metric_block("checkpoint194", baseline)
    print_required_metric_block("melhor safe 196", best_safe_summary)
    print_required_metric_block("melhor balanced 196", best_balanced_summary)
    print(f"\nResumo: {OUTPUT_DIR / 'tll196_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
