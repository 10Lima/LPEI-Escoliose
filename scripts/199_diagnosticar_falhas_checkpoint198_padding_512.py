import csv
import importlib.util
import os
from pathlib import Path

from config import get_dataset_dir
from metrics_common_padding_512 import TARGETS, safe_int


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
ACTIVE_DIR = PROCESSED_DIR / "cobb_results" / "baseline_2000" / "multicobb" / "active"

CHECKPOINT198_DIR = Path(os.environ.get(
    "SPINAL_199_CHECKPOINT198_DIR",
    str(ACTIVE_DIR / "checkpoint198_checkpoint194_tll197_v1"),
))
ORACLE187_DIR = Path(os.environ.get(
    "SPINAL_199_ORACLE187_DIR",
    str(ACTIVE_DIR / "oracle_global_checkpoint186_v1"),
))
OUTPUT_SUBDIR = Path(os.environ.get(
    "SPINAL_199_OUTPUT_SUBDIR",
    str(Path("baseline_2000") / "multicobb" / "active" / "diagnosticar_falhas_checkpoint198_v1"),
))
OUTPUT_DIR = PROCESSED_DIR / "cobb_results" / OUTPUT_SUBDIR
DIAG195_PATH = DATASET_DIR / "Scripts" / "195_diagnosticar_falhas_checkpoint194_padding_512.py"


def load_diag195_module():
    spec = importlib.util.spec_from_file_location("diag195_module", DIAG195_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar modulo: {DIAG195_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def count_lookup(count_rows):
    return {(row["metric"], row["group"]): safe_int(row["count"]) for row in count_rows}


def write_summary(count_rows, priority):
    counts = count_lookup(count_rows)
    lines = [
        "# Diagnostico 199 falhas checkpoint198",
        "",
        "## Entradas",
        "",
        f"- Checkpoint198: `{CHECKPOINT198_DIR}`.",
        f"- Oracle187: `{ORACLE187_DIR}`.",
        "",
        "## Resumo",
        "",
        f"- Falhas >5: {counts.get(('fail_images_gt5', 'all'), 0)}.",
        f"- Severos >8: {counts.get(('severe_images_gt8', 'all'), 0)}.",
        f"- Targets falhados >5: {counts.get(('failed_targets_gt5', 'all'), 0)}.",
        f"- Targets severos >8: {counts.get(('severe_targets_gt8', 'all'), 0)}.",
        "",
        "## Falhas por target",
        "",
    ]
    for target in TARGETS:
        lines.append(
            f"- {target}: {counts.get(('failed_targets_by_target', target), 0)} falhas target; "
            f"oracle_total<=5 em {counts.get(('failed_targets_oracle_total_le5', target), 0)}; "
            f"oracle_shortlist<=5 em {counts.get(('failed_targets_oracle_shortlist_le5', target), 0)}."
        )
    lines.extend(["", "## Dominancia em imagens falhadas", ""])
    for row in count_rows:
        if row["metric"] == "fail_images_by_dominant":
            lines.append(f"- {row['group']}: {row['count']}")
    lines.extend(["", "## Causas principais em imagens falhadas", ""])
    for row in count_rows:
        if row["metric"] == "fail_images_by_main_cause":
            lines.append(f"- {row['group']}: {row['count']}")
    lines.extend(["", "## Top prioridades", ""])
    for row in priority[:15]:
        lines.append(
            f"- {row['file_id']} {row['target']}: erro {row['abs_error']}, "
            f"oracle_total {row['oracle_total_abs_error']}, causa {row['inferred_cause']}, "
            f"score {row['priority_score']}."
        )
    lines.extend(["", "## Leitura", ""])
    target_counts = {
        target: counts.get(("failed_targets_by_target", target), 0)
        for target in TARGETS
    }
    next_target = max(target_counts, key=target_counts.get)
    lines.append(f"- Maior frente por targets falhados: {next_target}.")
    lines.append("- A decisao seguinte deve ponderar tambem dominancia em imagens falhadas e severos.")
    (OUTPUT_DIR / "diagnostico199_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_checkpoint198_rows(image_rows, target_rows):
    normalized_images = []
    for row in image_rows:
        item = dict(row)
        item["pt_source194"] = item.get("pt_source198", "")
        item["mt_source194"] = item.get("mt_source198", "")
        item["tll_source194"] = item.get("tll_source198", "")
        item["changed_targets194"] = item.get("changed_targets198", "")
        normalized_images.append(item)

    normalized_targets = []
    for row in target_rows:
        item = dict(row)
        item["source194"] = item.get("source198", "")
        normalized_targets.append(item)
    return normalized_images, normalized_targets


def main():
    prepare_output_dir()
    diag195 = load_diag195_module()
    image_rows = diag195.read_csv(CHECKPOINT198_DIR / "checkpoint198_image_rows.csv")
    target_rows = diag195.read_csv(CHECKPOINT198_DIR / "checkpoint198_target_rows.csv")
    oracle_rows = diag195.read_csv(ORACLE187_DIR / "oracle187_target_diagnostic_rows.csv")
    image_rows, target_rows = normalize_checkpoint198_rows(image_rows, target_rows)
    image_audit, target_audit = diag195.build_audit_rows(image_rows, target_rows, oracle_rows)
    count_rows = diag195.summarize_counts(image_audit, target_audit)
    priorities = diag195.priority_rows(target_audit)

    write_csv(OUTPUT_DIR / "diagnostico199_image_rows.csv", image_audit, diag195.union_fieldnames(image_audit))
    write_csv(OUTPUT_DIR / "diagnostico199_target_rows.csv", target_audit, diag195.union_fieldnames(target_audit))
    write_csv(OUTPUT_DIR / "diagnostico199_count_summary.csv", count_rows, diag195.union_fieldnames(count_rows))
    write_csv(OUTPUT_DIR / "diagnostico199_priority_targets.csv", priorities, diag195.union_fieldnames(priorities))
    write_summary(count_rows, priorities)

    print("\n===== DIAGNOSTICO199 FALHAS CHECKPOINT198 =====")
    for row in count_rows:
        if row["metric"] in {
            "fail_images_gt5", "severe_images_gt8", "failed_targets_gt5", "severe_targets_gt8",
            "failed_targets_by_target", "failed_targets_oracle_total_le5",
            "failed_targets_oracle_shortlist_le5",
        }:
            print(f"{row['metric']} | {row['group']}: {row['count']}")
    print(f"\nResumo: {OUTPUT_DIR / 'diagnostico199_summary.md'}")
    print("===== CONCLUIDO =====")


if __name__ == "__main__":
    main()
