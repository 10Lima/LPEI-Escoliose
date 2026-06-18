import csv
import importlib.util
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import TARGETS, metric_summary_from_arrays, round_safe, safe_float


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"
COMPARISON_DIR = (
    PROCESSED_DIR
    / "cobb_results"
    / "comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1"
)
SCRIPT_248_PATH = DATASET_DIR / "Scripts" / "248_aplicar_pipeline_refinada_final_com_candidatos_regenerados.py"
OVERWRITE = os.environ.get("SPINAL_250_OVERWRITE", "0").strip() == "1"


HISTORICAL_STAGES = [
    {
        "stage": "BASE186",
        "path": ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1" / "selector186_best_target_rows.csv",
        "reconstructed_stage": "BASE_DIRECT",
        "notes": "Base historica checkpoint186/selector186 vs base direta derivada dos candidatos regenerados.",
    },
    {
        "stage": "MT190",
        "path": ACTIVE_DIR / "checkpoint191_checkpoint186_mt190_v1" / "checkpoint191_target_rows.csv",
        "reconstructed_stage": "MT190",
        "notes": "Promocao 191 apos MT190.",
    },
    {
        "stage": "PT193",
        "path": ACTIVE_DIR / "checkpoint194_checkpoint191_pt193_v1" / "checkpoint194_target_rows.csv",
        "reconstructed_stage": "PT193",
        "notes": "Promocao 194 apos PT193.",
    },
    {
        "stage": "TLL197",
        "path": ACTIVE_DIR / "checkpoint198_checkpoint194_tll197_v1" / "checkpoint198_target_rows.csv",
        "reconstructed_stage": "TLL197",
        "notes": "Promocao 198 apos TL/L 196/197.",
    },
    {
        "stage": "PT200",
        "path": ACTIVE_DIR / "checkpoint201_checkpoint198_pt200_v1" / "checkpoint201_target_rows.csv",
        "reconstructed_stage": "PT200",
        "notes": "Promocao 201 apos PT200.",
    },
    {
        "stage": "SF204",
        "path": ACTIVE_DIR / "checkpoint205_checkpoint201_sf204_v1" / "checkpoint205_target_rows.csv",
        "reconstructed_stage": "SF204",
        "notes": "Promocao 205 apos severe-first 204.",
    },
]

ARTIFACTS = [
    {
        "stage": "MT190",
        "historical": ACTIVE_DIR / "validar_selector_mt_diversidade_checkpoint114_global_top1200_v1" / "selector125_oof_candidate_predictions.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "mt_candidates.csv",
        "equivalence": "not_equivalent",
        "reason": "O regenerado 245 nao contem predicted_good_score OOF equivalente ao selector125; usa score geometrico/selection_score.",
        "priority": 1,
    },
    {
        "stage": "PT193/PT200",
        "historical": ACTIVE_DIR / "gate_pt_hard_negatives_checkpoint114_v1" / "gate123_oof_candidate_predictions.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "pt_candidates.csv",
        "equivalence": "not_equivalent",
        "reason": "O regenerado 244 nao contem gate123_score/predicted_good_score OOF equivalentes; PT perde a semantica do gate historico.",
        "priority": 2,
    },
    {
        "stage": "TLL196",
        "historical": ACTIVE_DIR / "gate_two_stage_hard_negatives_tll_v1" / "gate114_oof_candidate_predictions.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "tll_candidates.csv",
        "equivalence": "not_equivalent",
        "reason": "O regenerado 246 nao separa o gate antigo old_TLL/gate114 com score OOF equivalente.",
        "priority": 3,
    },
    {
        "stage": "TLL196/TLL197",
        "historical": ACTIVE_DIR / "gerar_candidatos_tll_checkpoint156_relevantes_v1" / "tll159_candidate_rows.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "tll_candidates.csv",
        "equivalence": "partial",
        "reason": "O 246 tem candidatos TL/L novos, mas nao reproduz exatamente tll159 source/method/rank.",
        "priority": 4,
    },
    {
        "stage": "TLL196/TLL197",
        "historical": ACTIVE_DIR / "gerar_candidatos_tll_remaining_selector173_v1" / "tll175_candidate_rows.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "tll_candidates.csv",
        "equivalence": "partial",
        "reason": "O 246 tem variantes TL/L, mas nao reproduz exatamente tll175 source/method/rank.",
        "priority": 5,
    },
    {
        "stage": "SF204",
        "historical": ACTIVE_DIR / "preparar_severe_first_checkpoint201_v1" / "prep203_candidate_rows.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "severe_candidates.csv",
        "equivalence": "not_equivalent",
        "reason": "O 247 e derivado dos candidatos 244/246; nao reproduz prep203 construido a partir de PT193/TLL196 historicos.",
        "priority": 6,
    },
    {
        "stage": "SF204",
        "historical": ACTIVE_DIR / "validar_severe_first_pt_tll_checkpoint201_v1" / "sf204_best_safe_selected_cases.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "severe_candidates.csv",
        "equivalence": "not_equivalent",
        "reason": "O overlay severe do 248 nao tem o mesmo universo/foco de 25 severos do checkpoint201 historico.",
        "priority": 7,
    },
    {
        "stage": "BASE186",
        "historical": ACTIVE_DIR / "validar_f1_f2_shortlist_smoothing_checkpoint185_v1" / "selector186_best_target_rows.csv",
        "regenerated": COMPARISON_DIR / "train2000" / "candidates_valid" / "pt_case_summary.csv",
        "equivalence": "not_equivalent",
        "reason": "A base do 248 vem de Cobb direto por centerline/case summaries; nao materializa o checkpoint186 historico.",
        "priority": 8,
    },
    {
        "stage": "PT193",
        "historical": ACTIVE_DIR / "diagnosticar_falhas_checkpoint191_v1" / "diagnostico192_target_rows.csv",
        "regenerated": "",
        "equivalence": "missing",
        "reason": "Nao existe diagnostico192 equivalente por CENTERLINE_SUBDIR para definir foco PT dominante/falhado sem recorrer ao historico.",
        "priority": 9,
    },
    {
        "stage": "TLL196",
        "historical": ACTIVE_DIR / "diagnosticar_falhas_checkpoint194_v1" / "diagnostico195_target_rows.csv",
        "regenerated": "",
        "equivalence": "missing",
        "reason": "Nao existe diagnostico195 equivalente por CENTERLINE_SUBDIR para reproduzir foco TL/L historico.",
        "priority": 10,
    },
    {
        "stage": "PT200",
        "historical": ACTIVE_DIR / "diagnosticar_falhas_checkpoint198_v1" / "diagnostico199_target_rows.csv",
        "regenerated": "",
        "equivalence": "missing",
        "reason": "Nao existe diagnostico199 equivalente por CENTERLINE_SUBDIR para reproduzir o foco residual PT.",
        "priority": 11,
    },
]


def load_248():
    spec = importlib.util.spec_from_file_location("script248_module", SCRIPT_248_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar 248: {SCRIPT_248_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    if path.exists() and not OVERWRITE:
        raise FileExistsError(f"Output ja existe: {path}. Usa SPINAL_250_OVERWRITE=1 para substituir.")
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
    return fields or ["status"]


def rows_by_raw_id(rows):
    grouped = {}
    for row in rows:
        raw_id = row.get("raw_file_id", "")
        target = row.get("target", "")
        if target in TARGETS:
            grouped.setdefault(raw_id, {})[target] = row
    return {raw_id: values for raw_id, values in grouped.items() if all(target in values for target in TARGETS)}


def state_to_grouped(state, raw_ids):
    grouped = {}
    for file_id, values in state.items():
        raw_id = raw_ids.get(file_id, "")
        grouped[raw_id] = {
            target: {
                "raw_file_id": raw_id,
                "target": target,
                "gt": values[target]["gt"],
                "estimated": values[target]["estimated"],
                "source": values[target].get("source", ""),
            }
            for target in TARGETS
        }
    return grouped


def arrays(grouped, raw_ids):
    preds = []
    gts = []
    kept = []
    for raw_id in sorted(raw_ids):
        values = grouped.get(raw_id)
        if values is None:
            continue
        preds.append([safe_float(values[target]["estimated"]) for target in TARGETS])
        gts.append([safe_float(values[target]["gt"]) for target in TARGETS])
        kept.append(raw_id)
    return kept, np.asarray(preds, dtype=float), np.asarray(gts, dtype=float)


def clone_state(state):
    return {
        file_id: {target: dict(values[target]) for target in TARGETS}
        for file_id, values in state.items()
    }


def build_248_stage_groups(script248):
    state, raw_ids = script248.baseline_from_case_summaries("train2000")
    stages = {"BASE_DIRECT": state_to_grouped(clone_state(state), raw_ids)}

    mt_best, _read, _passed = script248.load_best_by_file(script248.candidate_path("train2000", "mt"), "MT", script248.passes_mt190)
    pt193_best, _read, _passed = script248.load_best_by_file(
        script248.candidate_path("train2000", "pt"), "PT", lambda row: script248.passes_pt_profile(row, script248.PT193_PROFILE)
    )
    tll196_best, _read, _passed = script248.load_best_by_file(
        script248.candidate_path("train2000", "tll"), "TL_L", lambda row: script248.passes_tll_profile(row, script248.TLL196_PROFILE)
    )
    pt200_best, _read, _passed = script248.load_best_by_file(
        script248.candidate_path("train2000", "pt"), "PT", lambda row: script248.passes_pt_profile(row, script248.PT200_PROFILE)
    )
    tll197_best, _read, _passed = script248.load_best_by_file(
        script248.candidate_path("train2000", "tll"), "TL_L", lambda row: script248.passes_tll_profile(row, script248.TLL197_PROFILE)
    )
    severe_best, _read, _passed = script248.load_best_severe_by_file(script248.candidate_path("train2000", "severe"))

    for stage_name, target, selections in [
        ("MT190", "MT", mt_best),
        ("PT193", "PT", pt193_best),
        ("TLL196", "TL_L", tll196_best),
        ("PT200", "PT", pt200_best),
        ("TLL197", "TL_L", tll197_best),
    ]:
        script248.apply_stage(state, selections, target, stage_name)
        stages[stage_name] = state_to_grouped(clone_state(state), raw_ids)
    script248.apply_severe(state, severe_best)
    stages["SF204"] = state_to_grouped(clone_state(state), raw_ids)
    return stages


def compare_stage(stage, historical, reconstructed, overlap_ids):
    kept_h, hist_preds, hist_gts = arrays(historical, overlap_ids)
    kept_r, recon_preds, recon_gts = arrays(reconstructed, overlap_ids)
    common = sorted(set(kept_h) & set(kept_r))
    _kept_h, hist_preds, hist_gts = arrays(historical, common)
    _kept_r, recon_preds, recon_gts = arrays(reconstructed, common)
    hist_metrics = metric_summary_from_arrays(hist_preds, hist_gts)
    recon_metrics = metric_summary_from_arrays(recon_preds, recon_gts, baseline_preds=hist_preds)
    abs_delta = np.abs(recon_preds - hist_preds)
    divergent_mask = np.any(abs_delta > 1e-4, axis=1)
    divergent_count = int(np.sum(divergent_mask))
    target_divergences = int(np.sum(abs_delta > 1e-4))
    max_delta = float(np.max(abs_delta)) if abs_delta.size else 0.0
    return {
        "stage": stage,
        "overlap_cases": len(common),
        "divergent_cases": divergent_count,
        "divergent_targets": target_divergences,
        "max_abs_estimate_delta": round_safe(max_delta),
        "historical_within_5": hist_metrics["within_5"],
        "reconstructed_within_5": recon_metrics["within_5"],
        "delta_within_5": round_safe(safe_float(recon_metrics["within_5"]) - safe_float(hist_metrics["within_5"])),
        "historical_mae3": hist_metrics["mae3"],
        "reconstructed_mae3": recon_metrics["mae3"],
        "delta_mae3": round_safe(safe_float(recon_metrics["mae3"]) - safe_float(hist_metrics["mae3"])),
        "historical_failures_gt5": hist_metrics["failures_gt5"],
        "reconstructed_failures_gt5": recon_metrics["failures_gt5"],
        "reconstructed_vs_historical_reg_target_gt5": recon_metrics["regressions_target_gt5"],
        "reconstructed_vs_historical_reg_img_gt1": recon_metrics["regressions_image_gt1"],
        "reconstructed_vs_historical_reg_img_gt3": recon_metrics["regressions_image_gt3"],
    }, common, hist_preds, recon_preds


def divergent_rows(stage, common, hist_grouped, recon_grouped, limit_per_stage=200):
    rows = []
    for raw_id in common:
        deltas = []
        for target in TARGETS:
            hist = safe_float(hist_grouped[raw_id][target]["estimated"])
            recon = safe_float(recon_grouped[raw_id][target]["estimated"])
            deltas.append(abs(recon - hist))
        if max(deltas) <= 1e-4:
            continue
        hist_errors = [
            abs(safe_float(hist_grouped[raw_id][target]["estimated"]) - safe_float(hist_grouped[raw_id][target]["gt"]))
            for target in TARGETS
        ]
        recon_errors = [
            abs(safe_float(recon_grouped[raw_id][target]["estimated"]) - safe_float(recon_grouped[raw_id][target]["gt"]))
            for target in TARGETS
        ]
        row = {
            "stage": stage,
            "raw_file_id": raw_id,
            "max_abs_estimate_delta": round_safe(max(deltas)),
            "hist_mae3": round_safe(float(np.mean(hist_errors))),
            "recon_mae3": round_safe(float(np.mean(recon_errors))),
            "delta_mae3": round_safe(float(np.mean(recon_errors) - np.mean(hist_errors))),
        }
        for idx, target in enumerate(TARGETS):
            row[f"{target}_hist"] = round_safe(hist_grouped[raw_id][target]["estimated"])
            row[f"{target}_recon"] = round_safe(recon_grouped[raw_id][target]["estimated"])
            row[f"{target}_delta"] = round_safe(safe_float(recon_grouped[raw_id][target]["estimated"]) - safe_float(hist_grouped[raw_id][target]["estimated"]))
            row[f"{target}_hist_source"] = hist_grouped[raw_id][target].get("source205", hist_grouped[raw_id][target].get("source", ""))
            row[f"{target}_recon_source"] = recon_grouped[raw_id][target].get("source", "")
        rows.append(row)
    rows.sort(key=lambda item: abs(safe_float(item["delta_mae3"])), reverse=True)
    return rows[:limit_per_stage]


def artifact_rows():
    rows = []
    for item in ARTIFACTS:
        hist = Path(item["historical"]) if item["historical"] else None
        regen = Path(item["regenerated"]) if item["regenerated"] else None
        rows.append({
            "priority": item["priority"],
            "stage": item["stage"],
            "historical_artifact": str(hist) if hist else "",
            "historical_exists": int(hist.exists()) if hist else 0,
            "regenerated_artifact": str(regen) if regen else "",
            "regenerated_exists": int(regen.exists()) if regen else 0,
            "equivalence": item["equivalence"],
            "reason": item["reason"],
        })
    return sorted(rows, key=lambda row: int(row["priority"]))


def dependency_graph_rows():
    return [
        {"step": "BASE186", "depends_on": "selector186_best_target_rows.csv", "regenerated_equivalent": "base_direct from 244/245/246 case summaries", "status": "not_equivalent"},
        {"step": "190 -> 191", "depends_on": "selector125_oof_candidate_predictions.csv", "regenerated_equivalent": "245 mt_candidates.csv", "status": "not_equivalent"},
        {"step": "193 -> 194", "depends_on": "diagnostico192 + gate123_oof_candidate_predictions.csv", "regenerated_equivalent": "244 pt_candidates.csv", "status": "not_equivalent"},
        {"step": "196 -> 197 -> 198", "depends_on": "diagnostico195 + gate114 + tll159 + tll175", "regenerated_equivalent": "246 tll_candidates.csv", "status": "partial_not_equivalent"},
        {"step": "200 -> 201", "depends_on": "diagnostico199 + gate123_oof_candidate_predictions.csv", "regenerated_equivalent": "244 pt_candidates.csv", "status": "not_equivalent"},
        {"step": "203 -> 204 -> 205", "depends_on": "audit202 + prep203_candidate_rows.csv", "regenerated_equivalent": "247 severe_candidates.csv", "status": "not_equivalent"},
    ]


def write_summary(step_rows, first_divergence, missing_rows):
    lines = [
        "# Auditoria 250 - paridade checkpoint205 vs 248 nos 1000 historicos",
        "",
        "## Primeira divergencia",
        "",
    ]
    if first_divergence:
        lines.append(
            f"- Primeira etapa divergente: `{first_divergence['stage']}` com "
            f"{first_divergence['divergent_cases']}/{first_divergence['overlap_cases']} casos divergentes."
        )
        lines.append(f"- Delta <=5 nessa etapa: {first_divergence['delta_within_5']} pontos.")
        lines.append(f"- Delta MAE3 nessa etapa: {first_divergence['delta_mae3']}.")
    else:
        lines.append("- Nenhuma divergencia encontrada.")
    lines.extend(["", "## Etapas", ""])
    for row in step_rows:
        lines.append(
            f"- {row['stage']}: divergentes {row['divergent_cases']}/{row['overlap_cases']}, "
            f"<=5 hist {row['historical_within_5']} vs recon {row['reconstructed_within_5']}, "
            f"MAE3 hist {row['historical_mae3']} vs recon {row['reconstructed_mae3']}."
        )
    lines.extend(["", "## 10 artefactos mais importantes em falta/nao equivalentes", ""])
    for row in missing_rows[:10]:
        lines.append(f"- {row['priority']}. `{Path(row['historical_artifact']).name if row['historical_artifact'] else row['stage']}`: {row['equivalence']}. {row['reason']}")
    lines.extend([
        "",
        "## Resposta",
        "",
        "Para que o 248 train2000 reproduza o checkpoint205 historico, faltam equivalentes fieis dos artefactos de candidatos, gates OOF e diagnosticos de foco usados pela cadeia 190-205.",
        "A primeira divergencia deve ser tratada antes de qualquer tentativa de melhorar metricas.",
    ])
    (COMPARISON_DIR / "250_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    script248 = load_248()
    reconstructed_stages = build_248_stage_groups(script248)
    checkpoint205_ids = set(rows_by_raw_id(read_csv(ACTIVE_DIR / "checkpoint205_checkpoint201_sf204_v1" / "checkpoint205_target_rows.csv")))

    step_rows = []
    divergent_all = []
    first_divergence = None
    for stage_info in HISTORICAL_STAGES:
        historical = rows_by_raw_id(read_csv(stage_info["path"]))
        reconstructed = reconstructed_stages[stage_info["reconstructed_stage"]]
        overlap = checkpoint205_ids & set(historical) & set(reconstructed)
        row, common, _hist_preds, _recon_preds = compare_stage(stage_info["stage"], historical, reconstructed, overlap)
        row["historical_path"] = str(stage_info["path"])
        row["reconstructed_stage"] = stage_info["reconstructed_stage"]
        row["notes"] = stage_info["notes"]
        step_rows.append(row)
        divergent_all.extend(divergent_rows(stage_info["stage"], common, historical, reconstructed))
        if first_divergence is None and int(row["divergent_cases"]) > 0:
            first_divergence = row

    missing = artifact_rows()
    first_rows = [first_divergence] if first_divergence else [{"status": "no_divergence"}]
    write_csv(COMPARISON_DIR / "250_step_parity_summary.csv", step_rows)
    write_csv(COMPARISON_DIR / "250_missing_artifacts.csv", missing)
    write_csv(COMPARISON_DIR / "250_divergent_cases.csv", divergent_all)
    write_csv(COMPARISON_DIR / "250_first_divergence.csv", first_rows)
    write_csv(COMPARISON_DIR / "250_artifact_dependency_graph.csv", dependency_graph_rows())
    write_summary(step_rows, first_divergence, missing)

    if first_divergence:
        print(
            f"Primeira divergencia: {first_divergence['stage']} "
            f"{first_divergence['divergent_cases']}/{first_divergence['overlap_cases']}"
        )
    for row in step_rows:
        print(
            f"{row['stage']}: divergentes={row['divergent_cases']} "
            f"<=5 hist={row['historical_within_5']} recon={row['reconstructed_within_5']}"
        )
    print(f"Resumo: {COMPARISON_DIR / '250_summary.md'}")


if __name__ == "__main__":
    main()
