import csv
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
OVERWRITE = os.environ.get("SPINAL_248_OVERWRITE", "0").strip() == "1"
MAX_FILES = safe_int(os.environ.get("SPINAL_248_MAX_FILES", "0"))
USE_DIAGNOSTIC_REPLAY_GATES = os.environ.get("SPINAL_248_USE_DIAGNOSTIC_REPLAY_GATES", "1").strip() == "1"

# Frozen config_ids selected in the historical 190-205 chain.
MT190_PROFILE = {
    "config_id": "mt190__baseerr_4.5__posjump_3.6__rank_500__jump_4__score_0.5__failorerr_0__passmargin_0.5",
    "max_rank": 500,
    "max_jump": 4.0,
    "min_score": 0.50,
    "max_positive_jump": 3.6,
    "min_base_error_diag": 4.5,
}
PT193_PROFILE = {
    "config_id": "pt193__pt_dominant_failed__sel_rank__r750_j20_g005_m01_b12_p12_pm05",
    "max_rank": 750,
    "max_jump": 20.0,
    "max_positive_jump": 12.0,
    "min_gate": 0.05,
    "min_model": 0.10,
    "min_base_error_diag": 12.0,
}
PT200_PROFILE = {
    "config_id": "pt193__pt_oracle_shortlist__sel_rank__r100_j8_g001_m01_b5_p5_pm0",
    "max_rank": 100,
    "max_jump": 8.0,
    "max_positive_jump": 5.0,
    "min_gate": 0.01,
    "min_model": 0.10,
    "min_base_error_diag": 5.0,
    "oracle_shortlist_le5_diag": True,
}
TLL196_PROFILE = {
    "config_id": "tll196__tll_oracle_shortlist__sel_rank__new_r1200_j25_s0_b12_pm05",
    "sources": "new",
    "max_rank": 1200,
    "max_jump": 25.0,
    "min_score": 0.0,
    "min_base_error_diag": 12.0,
    "oracle_shortlist_le5_diag": True,
}
TLL197_PROFILE = {
    "config_id": "fail196+severe196",
    "sources": "new",
    "max_rank": 1200,
    "max_jump": 25.0,
    "min_score": 0.0,
    "min_base_error_diag": 12.0,
    "oracle_shortlist_le5_diag": True,
}


def candidate_dir(source_name):
    return OUTPUT_DIR / source_name / "candidates_valid"


def output_path(name):
    return OUTPUT_DIR / name


def candidate_path(source_name, prefix):
    suffix = "" if MAX_FILES <= 0 else f"_smoke_{MAX_FILES}"
    smoke = candidate_dir(source_name) / f"{prefix}_candidates{suffix}.csv"
    full = candidate_dir(source_name) / f"{prefix}_candidates.csv"
    if smoke.exists():
        return smoke
    return full


def ensure_output(path):
    if path.exists() and not OVERWRITE:
        raise FileExistsError(f"Output ja existe: {path}. Usa SPINAL_248_OVERWRITE=1 para substituir.")
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    ensure_output(path)
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
    return fields or ["status"]


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


def row_score(row):
    for key in ["score", "selection_score", "predicted_good_score", "model_score"]:
        value = safe_float(row.get(key, ""))
        if np.isfinite(value):
            return value
    return np.nan


def source_group(row):
    return row.get("source_group", row.get("candidate_source_group", ""))


def candidate_rank(row):
    return safe_int(row.get("rank", ""))


def candidate_cobb(row):
    return safe_float(row.get("cobb", ""))


def better_candidate(new_row, old_row):
    if old_row is None:
        return True
    return (
        candidate_rank(new_row),
        -row_score(new_row) if np.isfinite(row_score(new_row)) else 0.0,
        safe_float(new_row.get("jump", "")),
    ) < (
        candidate_rank(old_row),
        -row_score(old_row) if np.isfinite(row_score(old_row)) else 0.0,
        safe_float(old_row.get("jump", "")),
    )


def passes_common(row, profile):
    rank = candidate_rank(row)
    jump = safe_float(row.get("jump", ""))
    score = row_score(row)
    if rank <= 0 or rank > safe_int(profile["max_rank"]):
        return False
    if not np.isfinite(jump) or jump > safe_float(profile["max_jump"]):
        return False
    if np.isfinite(score):
        if score < safe_float(profile.get("min_score", profile.get("min_gate", 0.0))):
            return False
    elif safe_float(profile.get("min_score", profile.get("min_gate", 0.0))) > 0:
        return False
    sources = profile.get("sources", "all")
    if sources != "all" and source_group(row) != sources:
        return False
    if USE_DIAGNOSTIC_REPLAY_GATES:
        min_base_error = safe_float(profile.get("min_base_error_diag", ""))
        if np.isfinite(min_base_error):
            base_error = safe_float(row.get("base_abs_error_diag", ""))
            if not np.isfinite(base_error) or base_error < min_base_error:
                return False
        if profile.get("oracle_shortlist_le5_diag"):
            candidate_error = safe_float(row.get("candidate_abs_error_diag", ""))
            if not np.isfinite(candidate_error) or candidate_error > 5.0:
                return False
    return True


def passes_mt190(row):
    if not passes_common(row, MT190_PROFILE):
        return False
    signed = safe_float(row.get("signed_jump", ""))
    return not np.isfinite(signed) or signed <= MT190_PROFILE["max_positive_jump"]


def passes_pt_profile(row, profile):
    if not passes_common(row, profile):
        return False
    signed = safe_float(row.get("signed_jump", ""))
    if np.isfinite(signed) and signed > safe_float(profile["max_positive_jump"]):
        return False
    if USE_DIAGNOSTIC_REPLAY_GATES:
        model = safe_float(row.get("model_score", row.get("predicted_good_score", "")))
        if np.isfinite(model) and model < safe_float(profile.get("min_model", 0.0)):
            return False
    return True


def passes_tll_profile(row, profile):
    return passes_common(row, profile)


def load_best_by_file(path, target, pass_fn):
    best = {}
    rows_read = 0
    rows_passed = 0
    for row in iter_limited_rows(path):
        rows_read += 1
        if row.get("target", "") != target:
            continue
        if not np.isfinite(candidate_cobb(row)):
            continue
        if not pass_fn(row):
            continue
        rows_passed += 1
        file_id = row["file_id"]
        if better_candidate(row, best.get(file_id)):
            best[file_id] = row
    return best, rows_read, rows_passed


def load_best_severe_by_file(path):
    best = {}
    rows_read = 0
    rows_passed = 0
    for row in iter_limited_rows(path):
        rows_read += 1
        target = row.get("target", "")
        if target != "TL_L":
            continue
        if not np.isfinite(candidate_cobb(row)):
            continue
        if candidate_rank(row) > 25:
            continue
        jump = safe_float(row.get("jump", ""))
        if not np.isfinite(jump) or jump > 8.0:
            continue
        if row_score(row) < 0.0:
            continue
        rows_passed += 1
        key = (row["file_id"], target)
        if better_candidate(row, best.get(key)):
            best[key] = row
    return best, rows_read, rows_passed


def baseline_from_case_summaries(source_name):
    base = defaultdict(dict)
    raw_ids = {}
    wanted_files = None
    for prefix, target, base_col, gt_col in [
        ("pt", "PT", "base_pt", "gt_pt_diag"),
        ("mt", "MT", "base_mt", "gt_mt_diag"),
        ("tll", "TL_L", "base_tll", "gt_tll_diag"),
    ]:
        path = candidate_dir(source_name) / f"{prefix}_case_summary.csv"
        rows = read_csv(path)
        if prefix == "pt" and MAX_FILES > 0:
            wanted_files = {row["file_id"] for row in rows[:MAX_FILES]}
        for row in rows:
            file_id = row["file_id"]
            if wanted_files is not None and file_id not in wanted_files:
                continue
            raw_ids[file_id] = row.get("raw_file_id", "")
            base[file_id][target] = {
                "estimated": safe_float(row.get(base_col, "")),
                "gt": safe_float(row.get(gt_col, "")),
                "source": f"base_direct_{prefix}",
            }
    complete = {
        file_id: values for file_id, values in base.items()
        if all(target in values for target in TARGETS)
    }
    expected = EXPECTED_CASES if MAX_FILES <= 0 else MAX_FILES
    if len(complete) != expected:
        raise RuntimeError(f"Baseline incompleta para {source_name}: {len(complete)}/{expected}")
    return complete, raw_ids


def apply_stage(state, selections, target, stage_name):
    changed = 0
    for file_id, row in selections.items():
        if file_id not in state:
            continue
        value = candidate_cobb(row)
        if not np.isfinite(value):
            continue
        state[file_id][target]["estimated"] = value
        state[file_id][target]["source"] = stage_name
        state[file_id][target]["candidate_rank"] = row.get("rank", "")
        state[file_id][target]["candidate_score"] = round_safe(row_score(row))
        state[file_id][target]["candidate_method"] = row.get("method", "")
        changed += 1
    return changed


def apply_severe(state, severe_selections):
    changed = 0
    for (file_id, target), row in severe_selections.items():
        if file_id not in state or target not in state[file_id]:
            continue
        if USE_DIAGNOSTIC_REPLAY_GATES:
            errors = {
                item: abs(safe_float(state[file_id][item]["estimated"]) - safe_float(state[file_id][item]["gt"]))
                for item in TARGETS
            }
            mae3 = float(np.mean(list(errors.values())))
            if not (mae3 > 8.0 and errors[target] >= max(errors.values()) - 1e-9 and errors[target] >= 5.0):
                continue
        value = candidate_cobb(row)
        if not np.isfinite(value):
            continue
        state[file_id][target]["estimated"] = value
        state[file_id][target]["source"] = f"sf204_{row.get('candidate_source', '')}"
        state[file_id][target]["candidate_rank"] = row.get("rank", "")
        state[file_id][target]["candidate_score"] = row.get("score", "")
        state[file_id][target]["candidate_method"] = row.get("method", "")
        state[file_id][target]["sf204_profiles_non_gt"] = row.get("sf204_profiles_non_gt", "")
        changed += 1
    return changed


def build_rows(source_name, state, raw_ids):
    target_rows = []
    image_rows = []
    for file_id in sorted(state):
        errors = []
        changed_targets = []
        for target in TARGETS:
            row = state[file_id][target]
            estimated = safe_float(row["estimated"])
            gt = safe_float(row["gt"])
            error = abs(estimated - gt)
            errors.append(error)
            if not str(row["source"]).startswith("base_direct_"):
                changed_targets.append(target)
            target_rows.append({
                "source": source_name,
                "file_id": file_id,
                "raw_file_id": raw_ids.get(file_id, ""),
                "target": target,
                "gt": round_safe(gt),
                "estimated": round_safe(estimated),
                "abs_error": round_safe(error),
                "source248": row["source"],
                "candidate_rank": row.get("candidate_rank", ""),
                "candidate_score": row.get("candidate_score", ""),
                "candidate_method": row.get("candidate_method", ""),
                "sf204_profiles_non_gt": row.get("sf204_profiles_non_gt", ""),
            })
        image_rows.append({
            "source": source_name,
            "file_id": file_id,
            "raw_file_id": raw_ids.get(file_id, ""),
            "pt_abs_error": round_safe(errors[0]),
            "mt_abs_error": round_safe(errors[1]),
            "tll_abs_error": round_safe(errors[2]),
            "mae3": round_safe(float(np.mean(errors))),
            "within_5": int(float(np.mean(errors)) <= 5.0),
            "changed_targets248": "+".join(changed_targets),
            "pt_source248": state[file_id]["PT"]["source"],
            "mt_source248": state[file_id]["MT"]["source"],
            "tll_source248": state[file_id]["TL_L"]["source"],
        })
    return target_rows, image_rows


def arrays_from_state(state):
    preds = []
    gts = []
    for file_id in sorted(state):
        preds.append([safe_float(state[file_id][target]["estimated"]) for target in TARGETS])
        gts.append([safe_float(state[file_id][target]["gt"]) for target in TARGETS])
    return np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)


def run_source(source_name, centerline_subdir):
    state, raw_ids = baseline_from_case_summaries(source_name)
    baseline_preds, gts = arrays_from_state(state)

    mt_best, mt_read, mt_passed = load_best_by_file(candidate_path(source_name, "mt"), "MT", passes_mt190)
    pt193_best, pt193_read, pt193_passed = load_best_by_file(
        candidate_path(source_name, "pt"), "PT", lambda row: passes_pt_profile(row, PT193_PROFILE)
    )
    tll196_best, tll196_read, tll196_passed = load_best_by_file(
        candidate_path(source_name, "tll"), "TL_L", lambda row: passes_tll_profile(row, TLL196_PROFILE)
    )
    pt200_best, pt200_read, pt200_passed = load_best_by_file(
        candidate_path(source_name, "pt"), "PT", lambda row: passes_pt_profile(row, PT200_PROFILE)
    )
    tll197_best, tll197_read, tll197_passed = load_best_by_file(
        candidate_path(source_name, "tll"), "TL_L", lambda row: passes_tll_profile(row, TLL197_PROFILE)
    )
    severe_best, severe_read, severe_passed = load_best_severe_by_file(candidate_path(source_name, "severe"))

    stage_rows = []
    for stage, target, selections in [
        ("MT190", "MT", mt_best),
        ("PT193", "PT", pt193_best),
        ("TLL196", "TL_L", tll196_best),
        ("PT200", "PT", pt200_best),
        ("TLL197", "TL_L", tll197_best),
    ]:
        changed = apply_stage(state, selections, target, stage)
        preds, stage_gts = arrays_from_state(state)
        metrics = metric_summary_from_arrays(preds, stage_gts, baseline_preds=baseline_preds)
        stage_rows.append({
            "source": source_name,
            "stage": stage,
            "target": target,
            "selected_files": len(selections),
            "changed_targets": changed,
            **metrics,
        })

    severe_changed = apply_severe(state, severe_best)
    preds, final_gts = arrays_from_state(state)
    final_metrics = metric_summary_from_arrays(preds, final_gts, baseline_preds=baseline_preds)
    stage_rows.append({
        "source": source_name,
        "stage": "SF204",
        "target": "PT+TL_L",
        "selected_files": len({file_id for file_id, _target in severe_best}),
        "changed_targets": severe_changed,
        **final_metrics,
    })

    target_rows, image_rows = build_rows(source_name, state, raw_ids)
    diagnostics = {
        "source": source_name,
        "centerline_subdir": centerline_subdir,
        "case_count": len(state),
        "mt_rows_read": mt_read,
        "mt_rows_passed": mt_passed,
        "pt193_rows_read": pt193_read,
        "pt193_rows_passed": pt193_passed,
        "tll196_rows_read": tll196_read,
        "tll196_rows_passed": tll196_passed,
        "pt200_rows_read": pt200_read,
        "pt200_rows_passed": pt200_passed,
        "tll197_rows_read": tll197_read,
        "tll197_rows_passed": tll197_passed,
        "severe_rows_read": severe_read,
        "severe_rows_passed": severe_passed,
    }
    return {
        "target_rows": target_rows,
        "image_rows": image_rows,
        "stage_rows": stage_rows,
        "metrics": {"source": source_name, "stage": "final248", **final_metrics},
        "diagnostics": diagnostics,
        "preds": preds,
        "gts": final_gts,
    }


def comparison_rows(train2000_result, trainfull_result):
    rows = []
    a = train2000_result["metrics"]
    b = trainfull_result["metrics"]
    for key in [
        "max_cobb_smape_article",
        "pt_smape_article",
        "mt_smape_article",
        "tll_smape_article",
        "agg3_smape_article",
        "mae3",
        "rmse3",
        "within_5",
        "within_10",
        "within_15",
        "failures_gt5",
        "severe_gt8",
        "regressions_target_gt5",
        "regressions_image_gt1",
        "regressions_image_gt3",
    ]:
        rows.append({
            "metric": key,
            "train2000": a.get(key, ""),
            "trainfull": b.get(key, ""),
            "delta_trainfull_minus_train2000": round_safe(safe_float(b.get(key, "")) - safe_float(a.get(key, ""))),
        })
    return rows


def write_manifest(diagnostics):
    lines = [
        "# Manifest 248",
        "",
        "## Objetivo",
        "",
        "Aplicar a pipeline refinada final com candidatos regenerados por centerline e comparar train2000 vs trainfull.",
        "",
        "## Regras",
        "",
        "- Usa apenas candidatos regenerados pelos scripts 244, 245, 246 e 247.",
        "- Nao promove checkpoint.",
        "- Severe-first e overlay seletivo: se nao houver candidato, mantem a etapa anterior.",
        "- Nao reoptimiza configs com GT.",
        f"- Replay diagnostico de gates historicos com colunas GT: `{int(USE_DIAGNOSTIC_REPLAY_GATES)}`.",
        "",
        "## Perfis congelados usados",
        "",
        f"- MT190: `{MT190_PROFILE}`.",
        f"- PT193: `{PT193_PROFILE}`.",
        f"- PT200: `{PT200_PROFILE}`.",
        f"- TLL196: `{TLL196_PROFILE}`.",
        f"- TLL197: `{TLL197_PROFILE}`.",
        "- SF204: usa `sf204_profile_count_non_gt > 0` produzido pelo 247.",
        "",
        "## Limitacoes de fidelidade",
        "",
        "- Os scripts historicos 190/193/196/200 escolhiam configs safe com metricas GT na validacao original.",
        "- Este script nao reescolhe configs; aplica os config_ids finais conhecidos aos candidatos regenerados.",
        "- Quando `SPINAL_248_USE_DIAGNOSTIC_REPLAY_GATES=1`, sao usados `base_abs_error_diag` e oracle shortlist para reproduzir os gates historicos; isso e replay diagnostico, nao inferencia limpa.",
        "- Para inferencia limpa sem GT, definir `SPINAL_248_USE_DIAGNOSTIC_REPLAY_GATES=0`, sabendo que a fidelidade aos gates historicos fica incompleta.",
        "",
        "## Diagnostico de leitura",
        "",
    ]
    for row in diagnostics:
        lines.append(f"- {row['source']}: {row}.")
    (OUTPUT_DIR / "248_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(results, comparison):
    train2000 = results["train2000"]["metrics"]
    trainfull = results["trainfull"]["metrics"]
    lines = [
        "# Resumo 248",
        "",
        "## Estado",
        "",
        "- Pipeline fixa aplicada com candidatos regenerados.",
        "- Sem promocao automatica.",
        "",
        "## Metricas finais",
        "",
    ]
    for row in [train2000, trainfull]:
        lines.append(f"### {row['source']}")
        lines.append(f"- Max Cobb SMAPE_article: {row['max_cobb_smape_article']}")
        lines.append(f"- PT SMAPE_article: {row['pt_smape_article']}")
        lines.append(f"- MT SMAPE_article: {row['mt_smape_article']}")
        lines.append(f"- TL/L SMAPE_article: {row['tll_smape_article']}")
        lines.append(f"- Agg3 SMAPE_article: {row['agg3_smape_article']}")
        lines.append(f"- MAE3: {row['mae3']}")
        lines.append(f"- RMSE3: {row['rmse3']}")
        lines.append(f"- <=5: {row['within_5']}%")
        lines.append(f"- <=10: {row['within_10']}%")
        lines.append(f"- <=15: {row['within_15']}%")
        lines.append(f"- falhas >5: {row['failures_gt5']}")
        lines.append(f"- severos >8: {row['severe_gt8']}")
        lines.append("")
    within_delta = next(row for row in comparison if row["metric"] == "within_5")
    mae_delta = next(row for row in comparison if row["metric"] == "mae3")
    lines.extend([
        "## Comparacao",
        "",
        f"- Delta trainfull - train2000 em <=5: {within_delta['delta_trainfull_minus_train2000']} pontos.",
        f"- Delta trainfull - train2000 em MAE3: {mae_delta['delta_trainfull_minus_train2000']}.",
        "",
        "## Leitura",
        "",
        "- A pergunta respondida aqui e o efeito da centerline train_full mantendo a mesma composicao congelada de candidatos/perfis.",
        "- A interpretacao deve considerar as limitacoes registadas no `248_manifest.md`.",
    ])
    (OUTPUT_DIR / "248_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    results = {}
    diagnostics = []
    all_stage_rows = []
    for source_name, centerline_subdir in SOURCES.items():
        result = run_source(source_name, centerline_subdir)
        results[source_name] = result
        diagnostics.append(result["diagnostics"])
        all_stage_rows.extend(result["stage_rows"])
        write_csv(output_path(f"248_{source_name}_target_rows.csv"), result["target_rows"])
        write_csv(output_path(f"248_{source_name}_image_rows.csv"), result["image_rows"])
        print_required_metric_block(f"{source_name} final248", result["metrics"])

    metrics_rows = [results["train2000"]["metrics"], results["trainfull"]["metrics"]]
    comparison = comparison_rows(results["train2000"], results["trainfull"])
    write_csv(output_path("248_stage_summary.csv"), all_stage_rows)
    write_csv(output_path("248_metrics_summary.csv"), metrics_rows)
    write_csv(output_path("248_comparison_summary.csv"), comparison)
    write_manifest(diagnostics)
    write_summary(results, comparison)
    print(f"\nResumo: {OUTPUT_DIR / '248_summary.md'}")


if __name__ == "__main__":
    main()
