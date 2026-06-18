import csv
from pathlib import Path

from config import get_dataset_dir


DATASET_DIR = get_dataset_dir()
PROCESSED_DIR = DATASET_DIR / "processed_padding_512"
CENTERLINES_DIR = PROCESSED_DIR / "centerlines"
OUTPUT_DIR = (
    PROCESSED_DIR
    / "cobb_results"
    / "comparacao_centerline_train2000_vs_trainfull_pipeline_refinada_final_v1"
)

SOURCES = {
    "train2000": "unet_baseline_2000_padding_512_centerline_val_all_3192",
    "trainfull": "unet_train_full_padding_512_centerline_val_all_3192",
}

TARGET_FILES = {
    "PT": "pt_candidates.csv",
    "MT": "mt_candidates.csv",
    "TL_L": "tll_candidates.csv",
    "SEVERE": "severe_candidates.csv",
}

EXPECTED_CASES = 3192


REGENERATION_AUDIT = [
    {
        "target": "PT",
        "historical_front": "193 -> 201 -> 211/229",
        "available_scripts": "207_gerar_candidatos_pt_severo_checkpoint205_padding_512.py; 229_gerar_candidatos_validar_trainfull_checkpoint205_padding_512.py",
        "status": "blocked",
        "reason": (
            "O gerador 207 aceita SPINAL_207_CENTERLINE_SUBDIR, mas foi desenhado para PT severo "
            "sobre diagnostico206/checkpoint205 e nao cobre obrigatoriamente as 3192 imagens. "
            "Usa-lo como PT global violaria a validacao de cobertura."
        ),
        "needed_work": (
            "Extrair a logica de pares PT do 207 para uma funcao all-cases, mantendo janelas, smooth, "
            "min_point_distance, min_norm_span e ordenacao historicos."
        ),
    },
    {
        "target": "MT",
        "historical_front": "190 -> 191",
        "available_scripts": "190_validar_mt188_protegido_checkpoint186_padding_512.py; 125_validar_selector_mt_diversidade_checkpoint114_global_padding_512.py",
        "status": "blocked",
        "reason": (
            "O 190 consome MT_CANDIDATES_DIR antigo. O 125 e um validador/selector historico que depende "
            "de checkpoint93/gate114 e nao expoe um regenerador all-cases por CENTERLINE_SUBDIR."
        ),
        "needed_work": (
            "Isolar o gerador de pares MT historico, parametrizar CENTERLINE_SUBDIR e exportar candidatos "
            "para todas as 3192 imagens sem seleccionar por GT."
        ),
    },
    {
        "target": "TL_L",
        "historical_front": "196 -> 197 -> 198",
        "available_scripts": "159_gerar_candidatos_tll_checkpoint156_relevantes_padding_512.py; 175_gerar_candidatos_tll_remaining_selector173_padding_512.py; 196_validar_tll_focado_checkpoint194_padding_512.py",
        "status": "blocked",
        "reason": (
            "159 e 175 aceitam ou reutilizam stack de centerline, mas escolhem casos a partir de diagnosticos "
            "historicos parciais. O 196 consome TLL_OLD_CANDIDATES_DIR/TLL159_DIR/TLL175_DIR antigos."
        ),
        "needed_work": (
            "Criar wrapper TL/L all-cases que reproduza current/spline/oracle pair generation com os mesmos "
            "limites de rank, jump, score e fontes historicas, sem depender de diagnosticos parciais."
        ),
    },
    {
        "target": "SEVERE",
        "historical_front": "203 -> 204 -> 205",
        "available_scripts": "203_preparar_severe_first_checkpoint201_padding_512.py; 204_validar_severe_first_pt_tll_checkpoint201_padding_512.py",
        "status": "blocked",
        "reason": (
            "Severe-first depende dos candidatos PT e TL/L regenerados e do checkpoint201 reaplicado. "
            "Sem PT/TL_L all-cases validos, severe_candidates.csv nao pode ser fiel."
        ),
        "needed_work": "Gerar severe apenas depois de PT/TL_L e checkpoint201 reaplicado estarem disponiveis.",
    },
]


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def centerline_count(subdir):
    path = CENTERLINES_DIR / subdir
    if not path.exists():
        raise FileNotFoundError(f"Pasta de centerlines nao encontrada: {path}")
    files = sorted(path.glob("*_centerline.png"))
    if not files:
        raise FileNotFoundError(f"Nenhuma centerline encontrada em: {path}")
    return path, len(files)


def candidate_dir(source_name):
    return OUTPUT_DIR / source_name / "candidates"


def write_blocked_candidate_files(source_name):
    rows = []
    for item in REGENERATION_AUDIT:
        rows.append({
            "status": item["status"],
            "target": item["target"],
            "reason": item["reason"],
            "needed_work": item["needed_work"],
        })
    by_target = {row["target"]: row for row in rows}
    out_dir = candidate_dir(source_name)
    for target, filename in TARGET_FILES.items():
        write_csv(out_dir / filename, [by_target[target]])


def write_manifest(counts):
    lines = [
        "# Manifest 243",
        "",
        "## Objetivo",
        "",
        "Regenerar candidatos PT, MT, TL/L e severe-first por CENTERLINE_SUBDIR, sem criar selectors novos.",
        "",
        "## Regra de fidelidade",
        "",
        "- Regenerar candidatos e permitido.",
        "- Reutilizar candidatos antigos de outra centerline nao e permitido.",
        "- Usar candidatos parciais de diagnosticos antigos como cobertura global nao e permitido.",
        "- GT pode existir em colunas diagnosticas, mas nao pode entrar na selecao/inferencia.",
        "",
        "## Fontes verificadas",
        "",
    ]
    for source_name, info in counts.items():
        lines.append(f"- {source_name}: `{info['centerline_dir']}` ({info['centerline_count']} centerlines).")
    lines.extend(["", "## Auditoria dos regeneradores historicos", ""])
    for item in REGENERATION_AUDIT:
        lines.append(
            f"- {item['target']} ({item['historical_front']}): {item['status']}. "
            f"Scripts: `{item['available_scripts']}`. Motivo: {item['reason']}"
        )
    lines.extend([
        "",
        "## Decisao",
        "",
        "- Bloqueado por fidelidade.",
        "- Nenhum ficheiro de candidatos foi marcado como valido.",
        "- O 242 deve continuar bloqueado ate existirem candidatos all-cases regenerados e validados para PT, MT e TL/L.",
    ])
    (OUTPUT_DIR / "243_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(counts):
    coverage_rows = []
    for source_name, info in counts.items():
        for item in REGENERATION_AUDIT:
            coverage_rows.append({
                "source": source_name,
                "centerline_subdir": SOURCES[source_name],
                "centerline_count": info["centerline_count"],
                "target": item["target"],
                "candidate_file": str(candidate_dir(source_name) / TARGET_FILES[item["target"]]),
                "candidate_status": item["status"],
                "covered_images": 0,
                "expected_images": EXPECTED_CASES,
                "missing_images": EXPECTED_CASES,
                "valid_for_242": 0,
                "reason": item["reason"],
            })
    write_csv(OUTPUT_DIR / "243_coverage_summary.csv", coverage_rows)

    lines = [
        "# Resumo 243",
        "",
        "## Estado",
        "",
        "- Bloqueado antes da geracao final de candidatos.",
        "",
        "## Porque bloqueou",
        "",
        "Os candidatos historicos usados pela cadeia refinada final nao estao todos expostos como regeneradores all-cases por CENTERLINE_SUBDIR.",
        "Criar candidatos genericos agora repetiria o erro metodologico do 240/241.",
        "",
        "## Outputs escritos",
        "",
        "- `train2000/candidates/pt_candidates.csv`",
        "- `train2000/candidates/mt_candidates.csv`",
        "- `train2000/candidates/tll_candidates.csv`",
        "- `train2000/candidates/severe_candidates.csv`",
        "- `trainfull/candidates/pt_candidates.csv`",
        "- `trainfull/candidates/mt_candidates.csv`",
        "- `trainfull/candidates/tll_candidates.csv`",
        "- `trainfull/candidates/severe_candidates.csv`",
        "- `243_coverage_summary.csv`",
        "- `243_manifest.md`",
        "- `243_summary.md`",
        "",
        "## Proximo passo correto",
        "",
        "Refatorar primeiro os geradores historicos de candidatos para aceitarem CENTERLINE_SUBDIR e cobrirem 3192 imagens, mantendo configs antigas.",
    ]
    (OUTPUT_DIR / "243_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for source_name, subdir in SOURCES.items():
        path, count = centerline_count(subdir)
        counts[source_name] = {
            "centerline_dir": path,
            "centerline_count": count,
        }
        write_blocked_candidate_files(source_name)

    write_manifest(counts)
    write_summary(counts)

    raise RuntimeError(
        "243 bloqueado por fidelidade: faltam regeneradores historicos all-cases por CENTERLINE_SUBDIR. "
        f"Ver {OUTPUT_DIR / '243_summary.md'}"
    )


if __name__ == "__main__":
    main()
