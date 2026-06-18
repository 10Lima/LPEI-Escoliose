import csv
import itertools
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

CHECKPOINT201_DIR = Path(os.environ.get(
    "SPINAL_204_CHECKPOINT201_DIR",
    str(ACTIVE_DIR / "checkpoint201_checkpoint198_pt200_v1"),
))
PREP203_DIR = Path(os.environ.get(
    "SPINAL_204_PREP203_DIR",
    str(ACTIVE_DIR / "preparar_severe_first_checkpoint201_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_204_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "validar_severe_first_pt_tll_checkpoint201_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR


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
    image_rows = read_csv(CHECKPOINT201_DIR / "checkpoint201_image_rows.csv")
    target_rows = read_csv(CHECKPOINT201_DIR / "checkpoint201_target_rows.csv")
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
            "sources201": {target: by_file[file_id][target]["source201"] for target in TARGETS},
        }
    return contexts


def focus_sets():
    severe_rows = read_csv(PREP203_DIR / "prep203_severe25_rows.csv")
    all_severe = {row["file_id"] for row in severe_rows}
    pt_failed = {row["file_id"] for row in severe_rows if "PT" in row["failed_targets_gt5"].split("+")}
    tll_failed = {row["file_id"] for row in severe_rows if "TL_L" in row["failed_targets_gt5"].split("+")}
    pt_dominant = {row["file_id"] for row in severe_rows if safe_int(row["pt_dominant"]) == 1}
    tll_dominant = {row["file_id"] for row in severe_rows if safe_int(row["tll_dominant"]) == 1}
    multi_target = {row["file_id"] for row in severe_rows if safe_int(row["multi_target"]) == 1}
    return {
        "all_severe": all_severe,
        "pt_failed": pt_failed,
        "tll_failed": tll_failed,
        "pt_dominant": pt_dominant,
        "tll_dominant": tll_dominant,
        "multi_target": multi_target,
    }


def load_candidate_groups():
    groups = {}
    for row in read_csv(PREP203_DIR / "prep203_candidate_rows.csv"):
        file_id = row["file_id"]
        target = row["target"]
        item = {
            "file_id": file_id,
            "target": target,
            "source": row["source"],
            "source_group": row["source_group"],
            "method": row.get("method", ""),
            "rank": safe_int(row["rank"]),
            "score": safe_float(row["score"]),
            "model_score": safe_float(row.get("model_score", "")),
            "selection_score": safe_float(row.get("selection_score", "")),
            "cobb": safe_float(row["cobb"]),
            "jump": safe_float(row["jump"]),
            "signed_jump": safe_float(row["signed_jump"]),
            "base_abs_error": safe_float(row["base_abs_error"]),
            "candidate_abs_error_diag": safe_float(row["candidate_abs_error_diag"]),
        }
        groups.setdefault((file_id, target), []).append(item)
    for items in groups.values():
        items.sort(key=lambda item: (safe_int(item["rank"]), -safe_float(item["score"])))
    return groups


def profiles():
    return {
        "PT": [
            {"id": "pt_off", "max_rank": 0, "max_jump": 0.0, "max_positive_jump": 0.0, "min_score": np.inf, "min_model": np.inf, "min_base_error": np.inf},
            {"id": "pt_r25_j5_g001_m01_b8", "max_rank": 25, "max_jump": 5.0, "max_positive_jump": 5.0, "min_score": 0.01, "min_model": 0.10, "min_base_error": 8.0},
            {"id": "pt_r100_j8_g001_m01_b8", "max_rank": 100, "max_jump": 8.0, "max_positive_jump": 5.0, "min_score": 0.01, "min_model": 0.10, "min_base_error": 8.0},
            {"id": "pt_r750_j20_g005_m01_b12", "max_rank": 750, "max_jump": 20.0, "max_positive_jump": 12.0, "min_score": 0.05, "min_model": 0.10, "min_base_error": 12.0},
        ],
        "TL_L": [
            {"id": "tll_off", "sources": "all", "max_rank": 0, "max_jump": 0.0, "min_score": np.inf, "min_base_error": np.inf},
            {"id": "tll_old_r25_j8_s05_b5", "sources": "old", "max_rank": 25, "max_jump": 8.0, "min_score": 0.50, "min_base_error": 5.0},
            {"id": "tll_all_r25_j8_s0_b5", "sources": "all", "max_rank": 25, "max_jump": 8.0, "min_score": 0.0, "min_base_error": 5.0},
            {"id": "tll_new_r500_j16_s0_b8", "sources": "new", "max_rank": 500, "max_jump": 16.0, "min_score": 0.0, "min_base_error": 8.0},
            {"id": "tll_new_r1200_j25_s0_b8", "sources": "new", "max_rank": 1200, "max_jump": 25.0, "min_score": 0.0, "min_base_error": 8.0},
        ],
    }


def configs():
    rows = []
    prof = profiles()
    for focus_name, mode, pt_profile, tll_profile in itertools.product(
        ["all_severe", "multi_target", "pt_failed", "tll_failed", "pt_dominant", "tll_dominant"],
        ["rank", "score"],
        prof["PT"],
        prof["TL_L"],
    ):
        rows.append({
            "config_id": f"sf204__{focus_name}__sel_{mode}__{pt_profile['id']}__{tll_profile['id']}",
            "focus_name": focus_name,
            "selection_mode": mode,
            "PT": pt_profile,
            "TL_L": tll_profile,
        })
    return rows


def candidate_passes(item, context, profile):
    if safe_int(profile["max_rank"]) <= 0:
        return False
    if safe_int(item["rank"]) > safe_int(profile["max_rank"]):
        return False
    if safe_float(item["jump"]) > safe_float(profile["max_jump"]):
        return False
    if item["target"] == "PT":
        if safe_float(item["signed_jump"]) > safe_float(profile["max_positive_jump"]):
            return False
        if safe_float(item["model_score"]) < safe_float(profile["min_model"]):
            return False
    if item["target"] == "TL_L" and profile.get("sources", "all") != "all":
        if item["source_group"] != profile["sources"]:
            return False
    if safe_float(item["score"]) < safe_float(profile["min_score"]):
        return False
    idx = TARGETS.index(item["target"])
    if context["base_errors"][idx] < safe_float(profile["min_base_error"]):
        return False
    return True


def select_item(items, context, profile, mode):
    passed = [item for item in items if candidate_passes(item, context, profile)]
    if not passed:
        return None
    if mode == "rank":
        return min(passed, key=lambda item: (safe_int(item["rank"]), -safe_float(item["score"])))
    return max(passed, key=lambda item: (safe_float(item["score"]), -safe_int(item["rank"])))


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
            for target in ["PT", "TL_L"]:
                item = select_item(groups.get((file_id, target), []), context, config[target], config["selection_mode"])
                if item is not None:
                    est[TARGETS.index(target)] = item["cobb"]
                    selected[(file_id, target)] = item
        preds.append(est)
        gts.append(context["gt"])
    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    summary = metric_summary_from_arrays(preds, gts, baseline_preds=baseline_preds)
    base_mae = np.mean(np.abs(baseline_preds - gts), axis=1)
    new_mae = np.mean(np.abs(preds - gts), axis=1)
    target_delta = np.abs(preds - gts) - np.abs(baseline_preds - gts)
    summary.update({
        "config_id": config["config_id"],
        "focus_name": config["focus_name"],
        "selection_mode": config["selection_mode"],
        "pt_profile": config["PT"]["id"],
        "tll_profile": config["TL_L"]["id"],
        "focus_cases": len(focus_files),
        "selected_cases": len({key[0] for key in selected}),
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
        -safe_int(summary["severe_gt8"]),
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
        sources = dict(context["sources201"])
        changed = []
        if file_id in focus_files:
            for target in ["PT", "TL_L"]:
                item = select_item(groups.get((file_id, target), []), context, config[target], config["selection_mode"])
                if item is None:
                    continue
                idx = TARGETS.index(target)
                est[idx] = item["cobb"]
                sources[target] = f"sf204_{item['source']}"
                changed.append(target)
                selected_rows.append({
                    "file_id": file_id,
                    "raw_file_id": context["raw_file_id"],
                    "target": target,
                    "gt": round_safe(context["gt"][idx]),
                    "base": round_safe(context["base"][idx]),
                    "selected": round_safe(item["cobb"]),
                    "base_abs_error": round_safe(context["base_errors"][idx]),
                    "selected_abs_error": round_safe(abs(item["cobb"] - context["gt"][idx])),
                    "delta_abs_error": round_safe(abs(item["cobb"] - context["gt"][idx]) - context["base_errors"][idx]),
                    "rank": item["rank"],
                    "score": round_safe(item["score"]),
                    "source": item["source"],
                    "method": item["method"],
                    "jump": round_safe(item["jump"]),
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
            "pt_source204": sources["PT"],
            "mt_source204": sources["MT"],
            "tll_source204": sources["TL_L"],
            "changed_targets204": "+".join(changed),
        })
        for idx, target in enumerate(TARGETS):
            target_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "gt": round_safe(context["gt"][idx]),
                "estimated": round_safe(est[idx]),
                "abs_error": round_safe(errors[idx]),
                "source204": sources[target],
            })
    return selected_rows, image_rows, target_rows


def compact(row):
    keys = [
        "config_id", "mae3", "rmse3", "within_5", "within_10", "within_15",
        "failures_gt5", "severe_gt8", "regressions_target_gt5",
        "regressions_image_gt1", "regressions_image_gt3", "new_failures",
        "new_severes", "fixed_failures", "fixed_severes", "selected_cases",
        "selected_targets", "max_cobb_smape_article", "pt_smape_article",
        "mt_smape_article", "tll_smape_article", "agg3_smape_article",
    ]
    return {key: row.get(key, "") for key in keys}


def write_summary(baseline, best_safe):
    lines = [
        "# Validacao 204 severe-first PT/TL-L checkpoint201",
        "",
        "## Escopo",
        "",
        "- Apenas 25 severos do checkpoint201 podem receber substituicoes.",
        "- Seleccao usa rank/score/jump/fonte; GT fica apenas para metricas.",
        "- MT fica congelado.",
        "",
        "## Baseline checkpoint201",
        "",
    ]
    for key, value in compact({"config_id": "checkpoint201", **baseline}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor safe 204", ""])
    for key, value in compact(best_safe).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Leitura", ""])
    if safe_int(best_safe["severe_gt8"]) < safe_int(baseline["severe_gt8"]):
        lines.append("- Existe melhoria segura em severos face ao checkpoint201.")
    elif safe_int(best_safe["failures_gt5"]) < safe_int(baseline["failures_gt5"]):
        lines.append("- Existe melhoria segura em falhas, mas sem reduzir severos.")
    else:
        lines.append("- Nenhuma configuracao safe melhorou falhas ou severos.")
    lines.append("- Este script valida severe-first; nao promove checkpoint.")
    (OUTPUT_DIR / "sf204_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    contexts = load_contexts()
    focus = focus_sets()
    groups = load_candidate_groups()
    baseline_preds = base_array(contexts)
    gts = gt_array(contexts)
    baseline = metric_summary_from_arrays(baseline_preds, gts)

    rows = []
    best_safe = None
    best_safe_config = None
    for config in configs():
        summary = evaluate_config(contexts, groups, focus, config, baseline_preds)
        rows.append(summary)
        key = sort_safe(summary, baseline)
        if best_safe is None or key > best_safe[0]:
            best_safe = (key, summary)
            best_safe_config = config

    best_summary = best_safe[1]
    selected_rows, image_rows, target_rows = build_detail_rows(contexts, groups, focus, best_safe_config)
    write_csv(OUTPUT_DIR / "sf204_config_summary.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "sf204_best_safe_summary.csv", [best_summary], union_fieldnames([best_summary]))
    write_csv(OUTPUT_DIR / "sf204_best_safe_selected_cases.csv", selected_rows, union_fieldnames(selected_rows) if selected_rows else ["file_id"])
    write_csv(OUTPUT_DIR / "sf204_best_safe_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "sf204_best_safe_target_rows.csv", target_rows, union_fieldnames(target_rows))
    write_summary(baseline, best_summary)

    print("\n===== SF204 SEVERE-FIRST CHECKPOINT201 =====")
    print_required_metric_block("checkpoint201", baseline)
    print_required_metric_block("melhor safe 204", best_summary)
    print(f"\nResumo: {OUTPUT_DIR / 'sf204_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
