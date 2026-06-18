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

CHECKPOINT186_DIR = Path(os.environ.get(
    "SPINAL_190_CHECKPOINT186_DIR",
    str(ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1"),
))
MT_CANDIDATES_DIR = Path(os.environ.get(
    "SPINAL_190_MT_CANDIDATES_DIR",
    str(ACTIVE_DIR / "validar_selector_mt_diversidade_checkpoint114_global_top1200_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_190_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "validar_mt188_protegido_checkpoint186_v1"),
))

OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
TARGET = "MT"
BASE_CONFIG_ID = "188__pt_off__mt_r500_j16_s05__tll_off__sel_rank"


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
    image_rows = read_csv(CHECKPOINT186_DIR / "selector186_best_image_rows.csv")
    target_rows = read_csv(CHECKPOINT186_DIR / "selector186_best_target_rows.csv")
    target_by_file = target_index(target_rows)
    contexts = {}
    for image in image_rows:
        file_id = image["file_id"]
        values = target_by_file[file_id]
        gt = [safe_float(values[target]["gt"]) for target in TARGETS]
        estimated = [safe_float(values[target]["estimated"]) for target in TARGETS]
        errors = [abs(estimated[idx] - gt[idx]) for idx in range(3)]
        contexts[file_id] = {
            "file_id": file_id,
            "raw_file_id": image.get("raw_file_id", values["PT"].get("raw_file_id", "")),
            "gt": gt,
            "base": estimated,
            "base_errors": errors,
            "base_mae3": float(np.mean(errors)),
            "mt_source186": values["MT"].get("source186", ""),
        }
    return contexts


def base_candidate_passes(item):
    return (
        safe_int(item["rank"]) <= 500
        and safe_float(item["jump"]) <= 16.0
        and safe_float(item["score"]) >= 0.50
    )


def stream_mt_candidates(contexts):
    path = MT_CANDIDATES_DIR / "selector125_oof_candidate_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV de candidatos MT nao encontrado: {path}")
    groups = defaultdict(list)
    wanted = set(contexts.keys())
    print(f"A ler candidatos MT: {path}", flush=True)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["file_id"]
            if file_id not in wanted:
                continue
            context = contexts[file_id]
            cobb = safe_float(row["cobb"])
            item = {
                "file_id": file_id,
                "rank": safe_int(row["rank"]),
                "score": safe_float(row.get("predicted_good_score", row.get("selection_score", ""))),
                "selection_score": safe_float(row.get("selection_score", "")),
                "cobb": cobb,
                "label_abs_error": safe_float(row.get("label_abs_error", "")),
                "jump": abs(cobb - context["base"][1]),
                "signed_jump": cobb - context["base"][1],
                "base_mt_abs_error": context["base_errors"][1],
                "base_mae3": context["base_mae3"],
                "window_name": row.get("window_name", ""),
                "norm_y_top": safe_float(row.get("norm_y_top", "")),
                "norm_y_bottom": safe_float(row.get("norm_y_bottom", "")),
                "norm_span": safe_float(row.get("norm_span", "")),
                "angle_top": safe_float(row.get("angle_top", "")),
                "angle_bottom": safe_float(row.get("angle_bottom", "")),
                "point_distance": safe_float(row.get("point_distance", "")),
                "smooth": safe_float(row.get("smooth", "")),
                "direction_up": safe_float(row.get("direction_up", "")),
            }
            if base_candidate_passes(item):
                groups[file_id].append(item)
    for file_id in list(groups.keys()):
        groups[file_id].sort(key=lambda item: (safe_int(item["rank"]), -safe_float(item["score"])))
    return groups


def configs():
    rows = []
    for (
        min_base_mt_error,
        max_positive_jump,
        max_rank,
        max_jump,
        min_score,
        require_failure_or_mt_error,
        block_pass_margin,
    ) in itertools.product(
        [0.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        [2.5, 3.0, 3.3, 3.6, 4.0, 16.0],
        [25, 100, 250, 500],
        [3.0, 3.5, 4.0, 5.0, 8.0, 16.0],
        [0.50, 0.505, 0.51, 0.52],
        [0, 1],
        [0.0, 0.1, 0.25, 0.5],
    ):
        if max_positive_jump > max_jump and max_jump < 16.0:
            continue
        rows.append({
            "config_id": (
                f"mt190__baseerr_{min_base_mt_error:g}__posjump_{max_positive_jump:g}"
                f"__rank_{max_rank}__jump_{max_jump:g}__score_{min_score:g}"
                f"__failorerr_{require_failure_or_mt_error}__passmargin_{block_pass_margin:g}"
            ),
            "min_base_mt_error": min_base_mt_error,
            "max_positive_jump": max_positive_jump,
            "max_rank": max_rank,
            "max_jump": max_jump,
            "min_score": min_score,
            "require_failure_or_mt_error": require_failure_or_mt_error,
            "block_pass_margin": block_pass_margin,
        })
    rows.append({
        "config_id": "mt190__off",
        "min_base_mt_error": np.inf,
        "max_positive_jump": 0.0,
        "max_rank": 0,
        "max_jump": 0.0,
        "min_score": np.inf,
        "require_failure_or_mt_error": 0,
        "block_pass_margin": 0.0,
    })
    return rows


def candidate_passes_guard(item, context, config):
    if safe_int(item["rank"]) > safe_int(config["max_rank"]):
        return False
    if safe_float(item["jump"]) > safe_float(config["max_jump"]):
        return False
    if safe_float(item["score"]) < safe_float(config["min_score"]):
        return False
    if safe_float(context["base_errors"][1]) < safe_float(config["min_base_mt_error"]):
        return False
    if safe_float(item["signed_jump"]) > safe_float(config["max_positive_jump"]):
        return False
    if safe_int(config["require_failure_or_mt_error"]):
        if not (context["base_mae3"] > 5.0 or context["base_errors"][1] >= config["min_base_mt_error"] + 1.0):
            return False
    margin = safe_float(config.get("block_pass_margin", 0.0))
    if margin > 0.0 and 5.0 - margin <= context["base_mae3"] <= 5.0:
        return False
    return True


def selected_item(items, context, config):
    if not items:
        return None
    item = items[0]
    return item if candidate_passes_guard(item, context, config) else None


def arrays_for_config(contexts, groups, config):
    preds = []
    gts = []
    selected = {}
    for file_id in sorted(contexts.keys()):
        context = contexts[file_id]
        est = list(context["base"])
        item = selected_item(groups.get(file_id, []), context, config)
        if item is not None:
            est[1] = item["cobb"]
            selected[file_id] = item
        preds.append(est)
        gts.append(context["gt"])
    return np.asarray(preds, dtype=float), np.asarray(gts, dtype=float), selected


def base_array(contexts):
    return np.asarray([contexts[file_id]["base"] for file_id in sorted(contexts.keys())], dtype=float)


def gt_array(contexts):
    return np.asarray([contexts[file_id]["gt"] for file_id in sorted(contexts.keys())], dtype=float)


def evaluate_config(contexts, groups, config, baseline_preds):
    preds, gts, selected = arrays_for_config(contexts, groups, config)
    summary = metric_summary_from_arrays(preds, gts, baseline_preds=baseline_preds)
    base_mae = np.mean(np.abs(baseline_preds - gts), axis=1)
    new_mae = np.mean(np.abs(preds - gts), axis=1)
    target_delta = np.abs(preds[:, 1] - gts[:, 1]) - np.abs(baseline_preds[:, 1] - gts[:, 1])
    summary.update({
        **config,
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


def sort_safe(summary, baseline_summary):
    return (
        safe_int(summary["regressions_target_gt5"]) == 0,
        safe_int(summary["regressions_image_gt3"]) == 0,
        safe_int(summary["regressions_image_gt1"]) == 0,
        safe_int(summary["new_severes"]) == 0,
        safe_int(summary["severe_gt8"]) <= safe_int(baseline_summary["severe_gt8"]),
        safe_int(summary["failures_gt5"]) <= safe_int(baseline_summary["failures_gt5"]),
        -safe_int(summary["failures_gt5"]),
        -safe_int(summary["severe_gt8"]),
        -safe_float(summary["mae3"]),
        safe_int(summary["selected_cases"]),
    )


def sort_balanced(summary, baseline_summary):
    return (
        safe_int(summary["regressions_target_gt5"]) == 0,
        safe_int(summary["regressions_image_gt3"]) == 0,
        safe_int(summary["new_severes"]) == 0,
        safe_int(summary["severe_gt8"]) <= safe_int(baseline_summary["severe_gt8"]),
        safe_int(summary["failures_gt5"]) < safe_int(baseline_summary["failures_gt5"]),
        -safe_int(summary["regressions_image_gt1"]),
        -safe_int(summary["failures_gt5"]),
        -safe_float(summary["mae3"]),
    )


def build_detail_rows(contexts, groups, config):
    selected_rows = []
    image_rows = []
    target_rows = []
    for file_id in sorted(contexts.keys()):
        context = contexts[file_id]
        est = list(context["base"])
        item = selected_item(groups.get(file_id, []), context, config)
        changed = []
        if item is not None:
            est[1] = item["cobb"]
            changed.append("MT")
        errors = np.abs(np.asarray(est) - np.asarray(context["gt"]))
        base_errors = np.asarray(context["base_errors"])
        if item is not None:
            selected_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": "MT",
                "gt": round_safe(context["gt"][1]),
                "base_mt": round_safe(context["base"][1]),
                "selected_mt": round_safe(item["cobb"]),
                "base_mt_abs_error": round_safe(base_errors[1]),
                "selected_mt_abs_error": round_safe(errors[1]),
                "delta_mt_abs_error": round_safe(errors[1] - base_errors[1]),
                "base_mae3": round_safe(context["base_mae3"]),
                "new_mae3": round_safe(np.mean(errors)),
                "delta_mae3": round_safe(np.mean(errors) - context["base_mae3"]),
                "rank": item["rank"],
                "score": round_safe(item["score"]),
                "jump": round_safe(item["jump"]),
                "signed_jump": round_safe(item["signed_jump"]),
                "window_name": item["window_name"],
                "norm_span": round_safe(item["norm_span"]),
                "norm_y_top": round_safe(item["norm_y_top"]),
                "norm_y_bottom": round_safe(item["norm_y_bottom"]),
                "config_id": config["config_id"],
            })
        image_rows.append({
            "file_id": file_id,
            "raw_file_id": context["raw_file_id"],
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(np.mean(errors)),
            "within_5": int(np.mean(errors) <= 5.0),
            "changed_targets190": "+".join(changed),
        })
        for idx, target in enumerate(TARGETS):
            target_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "gt": round_safe(context["gt"][idx]),
                "estimated": round_safe(est[idx]),
                "abs_error": round_safe(errors[idx]),
                "source190": "adaptive190_MT" if target == "MT" and item is not None else context.get("mt_source186", "") if target == "MT" else "checkpoint186",
            })
    return selected_rows, image_rows, target_rows


def compact(row):
    keys = [
        "config_id", "mae3", "rmse3", "within_5", "within_10", "within_15",
        "failures_gt5", "severe_gt8", "regressions_target_gt5",
        "regressions_image_gt1", "regressions_image_gt3", "new_severes",
        "fixed_failures", "selected_cases", "max_cobb_smape_article",
        "pt_smape_article", "mt_smape_article", "tll_smape_article",
        "agg3_smape_article",
    ]
    return {key: row.get(key, "") for key in keys}


def write_summary(baseline, best_safe, best_balanced):
    lines = [
        "# Validação 190 MT188 protegido checkpoint186",
        "",
        "## Escopo",
        "",
        f"- Base de candidatos: `{BASE_CONFIG_ID}`.",
        "- PT e TL/L ficam congelados no checkpoint186.",
        "- Guardas testados: erro MT base mínimo, salto positivo máximo, rank, jump, score e foco em falhas.",
        "- GT usado apenas para validação.",
        "",
        "## Baseline checkpoint186",
        "",
    ]
    for key, value in compact({"config_id": "checkpoint186", **baseline}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor safe 190", ""])
    for key, value in compact(best_safe).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Melhor balanced 190", ""])
    for key, value in compact(best_balanced).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Leitura", ""])
    if safe_int(best_safe["failures_gt5"]) < safe_int(baseline["failures_gt5"]):
        lines.append("- Existe melhoria segura face ao checkpoint186.")
    else:
        lines.append("- Nenhuma configuração safe melhorou as falhas face ao checkpoint186.")
    lines.append("- Este script valida MT protegido; não promove checkpoint automaticamente.")
    (OUTPUT_DIR / "mt190_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    contexts = load_contexts()
    groups = stream_mt_candidates(contexts)
    baseline_preds = base_array(contexts)
    gts = gt_array(contexts)
    baseline_summary = metric_summary_from_arrays(baseline_preds, gts)

    rows = []
    best_safe = None
    best_balanced = None
    best_safe_config = None
    best_balanced_config = None
    for config in configs():
        summary = evaluate_config(contexts, groups, config, baseline_preds)
        rows.append(summary)
        safe_key = sort_safe(summary, baseline_summary)
        balanced_key = sort_balanced(summary, baseline_summary)
        if best_safe is None or safe_key > best_safe[0]:
            best_safe = (safe_key, summary)
            best_safe_config = config
        if best_balanced is None or balanced_key > best_balanced[0]:
            best_balanced = (balanced_key, summary)
            best_balanced_config = config

    best_safe_summary = best_safe[1]
    best_balanced_summary = best_balanced[1]
    selected_rows, image_rows, target_rows = build_detail_rows(contexts, groups, best_safe_config)
    balanced_selected, balanced_images, balanced_targets = build_detail_rows(contexts, groups, best_balanced_config)

    write_csv(OUTPUT_DIR / "mt190_config_summary.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "mt190_best_safe_summary.csv", [best_safe_summary], union_fieldnames([best_safe_summary]))
    write_csv(OUTPUT_DIR / "mt190_best_balanced_summary.csv", [best_balanced_summary], union_fieldnames([best_balanced_summary]))
    write_csv(OUTPUT_DIR / "mt190_best_safe_selected_cases.csv", selected_rows, union_fieldnames(selected_rows) if selected_rows else ["file_id"])
    write_csv(OUTPUT_DIR / "mt190_best_safe_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "mt190_best_safe_target_rows.csv", target_rows, union_fieldnames(target_rows))
    write_csv(OUTPUT_DIR / "mt190_best_balanced_selected_cases.csv", balanced_selected, union_fieldnames(balanced_selected) if balanced_selected else ["file_id"])
    write_csv(OUTPUT_DIR / "mt190_best_balanced_image_rows.csv", balanced_images, union_fieldnames(balanced_images))
    write_csv(OUTPUT_DIR / "mt190_best_balanced_target_rows.csv", balanced_targets, union_fieldnames(balanced_targets))
    write_summary(baseline_summary, best_safe_summary, best_balanced_summary)

    print("\n===== MT190 PROTEGIDO CHECKPOINT186 =====")
    print_required_metric_block("checkpoint186", baseline_summary)
    print_required_metric_block("melhor safe 190", best_safe_summary)
    print_required_metric_block("melhor balanced 190", best_balanced_summary)
    print(f"\nResumo: {OUTPUT_DIR / 'mt190_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
