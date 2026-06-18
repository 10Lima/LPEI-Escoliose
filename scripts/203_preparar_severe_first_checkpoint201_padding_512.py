import csv
import importlib.util
import os
from pathlib import Path

import numpy as np

from config import get_dataset_dir
from metrics_common_padding_512 import TARGETS, round_safe, safe_float, safe_int


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT201_DIR = Path(os.environ.get(
    "SPINAL_203_CHECKPOINT201_DIR",
    str(ACTIVE_DIR / "checkpoint201_checkpoint198_pt200_v1"),
))
AUDIT202_DIR = Path(os.environ.get(
    "SPINAL_203_AUDIT202_DIR",
    str(ACTIVE_DIR / "auditar_gargalo_checkpoint201_v1"),
))
PT193_PATH = DATASET_DIR / "Scripts" / "193_validar_pt_focado_checkpoint191_padding_512.py"
TLL196_PATH = DATASET_DIR / "Scripts" / "196_validar_tll_focado_checkpoint194_padding_512.py"
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_203_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "preparar_severe_first_checkpoint201_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
MAX_ROWS_PER_FILE_PER_SOURCE = safe_int(os.environ.get("SPINAL_203_MAX_ROWS_PER_FILE_PER_SOURCE", "1200"))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar modulo: {path}")
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


def load_contexts(severe_files):
    image_rows = read_csv(CHECKPOINT201_DIR / "checkpoint201_image_rows.csv")
    target_rows = read_csv(CHECKPOINT201_DIR / "checkpoint201_target_rows.csv")
    by_file = target_index(target_rows)
    contexts = {}
    for image in image_rows:
        file_id = image["file_id"]
        if file_id not in severe_files:
            continue
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
        }
    return contexts


def pt_candidates(pt193, contexts):
    groups = pt193.stream_pt_candidates(contexts, set(contexts))
    rows = []
    for file_id, items in groups.items():
        context = contexts[file_id]
        for item in items:
            rows.append({
                "file_id": file_id,
                "raw_file_id": context["raw_file_id"],
                "target": "PT",
                "source": "pt_gate123",
                "source_group": "pt",
                "method": item.get("window_name", ""),
                "rank": item["rank"],
                "score": round_safe(item.get("gate123_score", "")),
                "model_score": round_safe(item.get("predicted_good_score", "")),
                "selection_score": round_safe(item.get("selection_score", "")),
                "cobb": round_safe(item["cobb"]),
                "jump": round_safe(item["jump"]),
                "signed_jump": round_safe(item["signed_jump"]),
                "base_abs_error": round_safe(context["base_errors"][0]),
                "candidate_abs_error_diag": round_safe(abs(item["cobb"] - context["gt"][0])),
                "delta_abs_error_diag": round_safe(abs(item["cobb"] - context["gt"][0]) - context["base_errors"][0]),
                "norm_span": round_safe(item.get("norm_span", "")),
                "norm_y_top": round_safe(item.get("norm_y_top", "")),
                "norm_y_bottom": round_safe(item.get("norm_y_bottom", "")),
            })
    return rows


def first_score(row, cols):
    for col in cols:
        value = safe_float(row.get(col, ""))
        if np.isfinite(value):
            return value
    return np.nan


def tll_specs(tll196):
    return tll196.candidate_specs()


def tll_candidates(tll196, contexts):
    rows = []
    wanted = set(contexts)
    for spec in tll_specs(tll196):
        path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"CSV TL/L nao encontrado: {path}")
        per_file_counts = {}
        print(f"A ler candidatos {spec['source']}: {path}", flush=True)
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row.get("file_id", "")
                if file_id not in wanted:
                    continue
                count = per_file_counts.get(file_id, 0)
                if count >= MAX_ROWS_PER_FILE_PER_SOURCE:
                    continue
                cobb = safe_float(row.get("cobb", ""))
                if not np.isfinite(cobb):
                    continue
                per_file_counts[file_id] = count + 1
                context = contexts[file_id]
                score = first_score(row, spec["score_cols"])
                rank = safe_int(row.get(spec["rank_col"], ""))
                rows.append({
                    "file_id": file_id,
                    "raw_file_id": context["raw_file_id"],
                    "target": "TL_L",
                    "source": spec["source"],
                    "source_group": spec["source_group"],
                    "method": row.get(spec["method_col"], "") if spec["method_col"] else spec["source"],
                    "rank": rank,
                    "score": round_safe(score),
                    "model_score": "",
                    "selection_score": round_safe(score),
                    "cobb": round_safe(cobb),
                    "jump": round_safe(abs(cobb - context["base"][2])),
                    "signed_jump": round_safe(cobb - context["base"][2]),
                    "base_abs_error": round_safe(context["base_errors"][2]),
                    "candidate_abs_error_diag": round_safe(abs(cobb - context["gt"][2])),
                    "delta_abs_error_diag": round_safe(abs(cobb - context["gt"][2]) - context["base_errors"][2]),
                    "norm_span": round_safe(row.get("norm_span", "")),
                    "norm_y_top": round_safe(row.get("norm_y_top", "")),
                    "norm_y_bottom": round_safe(row.get("norm_y_bottom", "")),
                })
    return rows


def summarize_candidates(severe_rows, candidate_rows):
    rows = []
    by_key = {}
    for row in candidate_rows:
        by_key.setdefault((row["file_id"], row["target"]), []).append(row)
    for severe in severe_rows:
        file_id = severe["file_id"]
        for target in ["PT", "TL_L"]:
            items = by_key.get((file_id, target), [])
            good_le5 = [
                item for item in items
                if np.isfinite(safe_float(item["candidate_abs_error_diag"]))
                and safe_float(item["candidate_abs_error_diag"]) <= 5.0
            ]
            best = min(items, key=lambda item: safe_float(item["candidate_abs_error_diag"])) if items else {}
            rows.append({
                "file_id": file_id,
                "raw_file_id": severe.get("raw_file_id", ""),
                "target": target,
                "dominant_targets": severe.get("dominant_targets", ""),
                "failed_targets_gt5": severe.get("failed_targets_gt5", ""),
                "severe_targets_gt8": severe.get("severe_targets_gt8", ""),
                "candidate_count": len(items),
                "good_candidate_le5_diag": int(len(good_le5) > 0),
                "good_candidate_count_le5_diag": len(good_le5),
                "best_candidate_abs_error_diag": round_safe(best.get("candidate_abs_error_diag", "")),
                "best_candidate_rank": best.get("rank", ""),
                "best_candidate_source": best.get("source", ""),
                "best_candidate_method": best.get("method", ""),
                "best_candidate_jump": best.get("jump", ""),
            })
    return rows


def write_summary(severe_rows, candidate_rows, case_summary):
    pt_good = sum(safe_int(row["good_candidate_le5_diag"]) == 1 for row in case_summary if row["target"] == "PT")
    tll_good = sum(safe_int(row["good_candidate_le5_diag"]) == 1 for row in case_summary if row["target"] == "TL_L")
    lines = [
        "# Preparacao 203 severe-first checkpoint201",
        "",
        "## Escopo",
        "",
        "- Usa apenas os 25 severos do checkpoint201.",
        "- Exporta candidatos PT e TL/L para validacao posterior.",
        "- Erro do candidato e marcado como diagnostico, nao deve ser usado na inferencia.",
        "",
        "## Resumo",
        "",
        f"- Severos: {len(severe_rows)}.",
        f"- Candidatos exportados: {len(candidate_rows)}.",
        f"- Severos com candidato PT <=5 diagnostico: {pt_good}.",
        f"- Severos com candidato TL/L <=5 diagnostico: {tll_good}.",
        "",
        "## Leitura",
        "",
        "- O proximo script deve selecionar por rank/score/jump/fonte, sem usar erro diagnostico.",
    ]
    (OUTPUT_DIR / "prep203_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    severe_rows = read_csv(AUDIT202_DIR / "audit202_severe25_rows.csv")
    severe_files = {row["file_id"] for row in severe_rows}
    contexts = load_contexts(severe_files)
    pt193 = load_module(PT193_PATH, "pt193_module")
    tll196 = load_module(TLL196_PATH, "tll196_module")

    candidates = []
    candidates.extend(pt_candidates(pt193, contexts))
    candidates.extend(tll_candidates(tll196, contexts))
    case_summary = summarize_candidates(severe_rows, candidates)

    write_csv(OUTPUT_DIR / "prep203_severe25_rows.csv", severe_rows, union_fieldnames(severe_rows))
    write_csv(OUTPUT_DIR / "prep203_candidate_rows.csv", candidates, union_fieldnames(candidates))
    write_csv(OUTPUT_DIR / "prep203_case_target_summary.csv", case_summary, union_fieldnames(case_summary))
    write_summary(severe_rows, candidates, case_summary)

    print("\n===== PREP203 SEVERE-FIRST CHECKPOINT201 =====")
    print(f"Severos: {len(severe_rows)}")
    print(f"Candidatos exportados: {len(candidates)}")
    print(f"Resumo: {OUTPUT_DIR / 'prep203_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
