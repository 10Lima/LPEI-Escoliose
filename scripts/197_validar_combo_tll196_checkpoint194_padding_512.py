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
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_197_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "validar_combo_tll196_checkpoint194_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR

TLL196_PATH = DATASET_DIR / "Scripts" / "196_validar_tll_focado_checkpoint194_padding_512.py"
FAIL_CONFIG_ID = "tll196__tll_oracle_shortlist__sel_rank__new_r1200_j25_s0_b12_pm05"
SEVERE_CONFIG_ID = "tll196__severe_tll_dominant__sel_rank__new_r1200_j25_s0_b12_pm05"
TARGET = "TL_L"
TARGET_IDX = TARGETS.index(TARGET)


def load_tll196_module():
    spec = importlib.util.spec_from_file_location("tll196_module", TLL196_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar modulo: {TLL196_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def config_by_id(module, config_id):
    for config in module.configs():
        if config["config_id"] == config_id:
            return config
    raise RuntimeError(f"Config nao encontrada: {config_id}")


def base_array(contexts):
    return np.asarray([contexts[file_id]["base"] for file_id in sorted(contexts)], dtype=float)


def gt_array(contexts):
    return np.asarray([contexts[file_id]["gt"] for file_id in sorted(contexts)], dtype=float)


def evaluate_combo(module, contexts, groups, focus, steps, baseline_preds):
    preds = []
    gts = []
    selected_by_file = {}
    selected_rows = []
    for file_id in sorted(contexts):
        context = contexts[file_id]
        est = list(context["base"])
        chosen = None
        chosen_step = ""
        for step_name, config in steps:
            if file_id not in focus[config["focus_name"]]:
                continue
            item = module.selected_item(groups.get(file_id, []), context, config)
            if item is None:
                continue
            chosen = item
            chosen_step = step_name
        if chosen is not None:
            est[TARGET_IDX] = chosen["cobb"]
            selected_by_file[file_id] = chosen
            selected_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": TARGET,
                "selection_step": chosen_step,
                "gt": round_safe(context["gt"][TARGET_IDX]),
                "base_tll": round_safe(context["base"][TARGET_IDX]),
                "selected_tll": round_safe(chosen["cobb"]),
                "base_tll_abs_error": round_safe(context["base_errors"][TARGET_IDX]),
                "selected_tll_abs_error": round_safe(abs(chosen["cobb"] - context["gt"][TARGET_IDX])),
                "delta_tll_abs_error": round_safe(abs(chosen["cobb"] - context["gt"][TARGET_IDX]) - context["base_errors"][TARGET_IDX]),
                "base_mae3": round_safe(context["base_mae3"]),
                "rank": chosen["rank"],
                "source": chosen["source"],
                "method": chosen["method"],
                "score": round_safe(chosen["score"]),
                "jump": round_safe(chosen["jump"]),
                "signed_jump": round_safe(chosen["signed_jump"]),
            })
        preds.append(est)
        gts.append(context["gt"])

    preds = np.asarray(preds, dtype=float)
    gts = np.asarray(gts, dtype=float)
    summary = metric_summary_from_arrays(preds, gts, baseline_preds=baseline_preds)
    base_mae = np.mean(np.abs(baseline_preds - gts), axis=1)
    new_mae = np.mean(np.abs(preds - gts), axis=1)
    target_delta = np.abs(preds[:, TARGET_IDX] - gts[:, TARGET_IDX]) - np.abs(baseline_preds[:, TARGET_IDX] - gts[:, TARGET_IDX])
    summary.update({
        "config_id": "+".join(step_name for step_name, _ in steps),
        "selected_cases": len(selected_by_file),
        "selected_targets": len(selected_by_file),
        "improved_targets": int(np.sum(target_delta < -1e-6)),
        "worsened_targets": int(np.sum(target_delta > 1e-6)),
        "fixed_failures": int(np.sum((base_mae > 5.0) & (new_mae <= 5.0))),
        "new_failures": int(np.sum((base_mae <= 5.0) & (new_mae > 5.0))),
        "fixed_severes": int(np.sum((base_mae > 8.0) & (new_mae <= 8.0))),
        "new_severes": int(np.sum((base_mae <= 8.0) & (new_mae > 8.0))),
        "mean_delta_mae3": round_safe(float(np.mean(new_mae - base_mae))),
    })
    return summary, selected_rows, preds, gts


def build_detail_rows(contexts, preds):
    image_rows = []
    target_rows = []
    for row_idx, file_id in enumerate(sorted(contexts)):
        context = contexts[file_id]
        est = preds[row_idx]
        errors = np.abs(est - np.asarray(context["gt"]))
        changed = abs(est[TARGET_IDX] - context["base"][TARGET_IDX]) > 1e-4
        sources = dict(context["sources194"])
        if changed:
            sources[TARGET] = "tll197_combo"
        image_rows.append({
            "file_id": file_id,
            "raw_file_id": context["raw_file_id"],
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(np.mean(errors)),
            "within_5": int(np.mean(errors) <= 5.0),
            "pt_source197": sources["PT"],
            "mt_source197": sources["MT"],
            "tll_source197": sources["TL_L"],
            "changed_targets197": TARGET if changed else "",
        })
        for idx, target in enumerate(TARGETS):
            target_rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": target,
                "gt": round_safe(context["gt"][idx]),
                "estimated": round_safe(est[idx]),
                "abs_error": round_safe(errors[idx]),
                "source197": sources[target],
            })
    return image_rows, target_rows


def sort_candidate(summary):
    return (
        safe_int(summary["regressions_target_gt5"]) == 0,
        safe_int(summary["regressions_image_gt3"]) == 0,
        safe_int(summary["regressions_image_gt1"]) == 0,
        safe_int(summary["new_failures"]) == 0,
        safe_int(summary["new_severes"]) == 0,
        -safe_int(summary["severe_gt8"]),
        -safe_int(summary["failures_gt5"]),
        -safe_float(summary["mae3"]),
    )


def write_summary(baseline, rows, best):
    lines = [
        "# Validacao 197 combo TL/L checkpoint194",
        "",
        "## Escopo",
        "",
        "- Combina os dois melhores candidatos TL/L do 196.",
        "- PT e MT ficam congelados no checkpoint194.",
        "- Testa ganho em falhas e severos antes de promover checkpoint.",
        "",
        "## Baseline checkpoint194",
        "",
        f"- MAE3: {baseline['mae3']}.",
        f"- <=5: {baseline['within_5']}.",
        f"- Falhas: {baseline['failures_gt5']}.",
        f"- Severos: {baseline['severe_gt8']}.",
        "",
        "## Configs testadas",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['config_id']}: MAE3 {row['mae3']}, <=5 {row['within_5']}, "
            f"falhas {row['failures_gt5']}, severos {row['severe_gt8']}, "
            f"reg target >5 {row['regressions_target_gt5']}, reg img >1 {row['regressions_image_gt1']}, "
            f"reg img >3 {row['regressions_image_gt3']}, novas falhas {row['new_failures']}, "
            f"novos severos {row['new_severes']}."
        )
    lines.extend([
        "",
        "## Melhor candidato",
        "",
        f"- config_id: {best['config_id']}.",
        f"- MAE3: {baseline['mae3']} -> {best['mae3']}.",
        f"- <=5: {baseline['within_5']} -> {best['within_5']}.",
        f"- Falhas: {baseline['failures_gt5']} -> {best['failures_gt5']}.",
        f"- Severos: {baseline['severe_gt8']} -> {best['severe_gt8']}.",
        f"- Regressões target >5: {best['regressions_target_gt5']}.",
        f"- Regressões imagem >1: {best['regressions_image_gt1']}.",
        f"- Regressões imagem >3: {best['regressions_image_gt3']}.",
        "",
        "## Leitura",
        "",
        "- Se o melhor candidato mantiver 0 regressões, pode ser promovido num checkpoint separado.",
    ])
    (OUTPUT_DIR / "tll197_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    module = load_tll196_module()
    contexts = module.load_contexts()
    focus = module.focus_sets()
    wanted_files = set().union(*focus.values())
    groups = module.stream_tll_candidates(contexts, wanted_files)
    baseline_preds = base_array(contexts)
    gts = gt_array(contexts)
    baseline = metric_summary_from_arrays(baseline_preds, gts)

    fail_config = config_by_id(module, FAIL_CONFIG_ID)
    severe_config = config_by_id(module, SEVERE_CONFIG_ID)
    combo_defs = [
        [("fail196", fail_config)],
        [("severe196", severe_config)],
        [("fail196", fail_config), ("severe196", severe_config)],
        [("severe196", severe_config), ("fail196", fail_config)],
    ]

    rows = []
    detail = {}
    for steps in combo_defs:
        summary, selected_rows, preds, _ = evaluate_combo(module, contexts, groups, focus, steps, baseline_preds)
        rows.append(summary)
        detail[summary["config_id"]] = (selected_rows, preds)

    best = max(rows, key=sort_candidate)
    best_selected, best_preds = detail[best["config_id"]]
    image_rows, target_rows = build_detail_rows(contexts, best_preds)

    write_csv(OUTPUT_DIR / "tll197_config_summary.csv", rows, union_fieldnames(rows))
    write_csv(OUTPUT_DIR / "tll197_best_summary.csv", [best], union_fieldnames([best]))
    write_csv(OUTPUT_DIR / "tll197_best_selected_cases.csv", best_selected, union_fieldnames(best_selected) if best_selected else ["file_id"])
    write_csv(OUTPUT_DIR / "tll197_best_image_rows.csv", image_rows, union_fieldnames(image_rows))
    write_csv(OUTPUT_DIR / "tll197_best_target_rows.csv", target_rows, union_fieldnames(target_rows))
    write_summary(baseline, rows, best)

    print("\n===== TLL197 COMBO CHECKPOINT194 =====")
    print_required_metric_block("checkpoint194", baseline)
    print_required_metric_block("melhor 197", best)
    print(f"\nResumo: {OUTPUT_DIR / 'tll197_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
