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

TRAIN2000_CENTERLINE_SUBDIR = "unet_baseline_2000_padding_512_centerline_val_all_3192"
TRAINFULL_CENTERLINE_SUBDIR = "unet_train_full_padding_512_centerline_val_all_3192"

TRAIN2000_DIR = OUTPUT_DIR / "train2000"
TRAINFULL_DIR = OUTPUT_DIR / "trainfull"
COMPARISON_DIR = OUTPUT_DIR / "comparison"


DIRECTLY_REUSABLE = [
    {
        "component": "Cobb direto base",
        "scripts": "232_validar_cobb_trainfull_val_all_3192_padding_512.py",
        "status": "reutilizavel_com_env",
        "notes": "Aceita SPINAL_232_CENTERLINE_SUBDIR.",
    },
    {
        "component": "PT severo checkpoint205",
        "scripts": "207_gerar_candidatos_pt_severo_checkpoint205_padding_512.py",
        "status": "reutilizavel_com_env",
        "notes": "Aceita SPINAL_207_CENTERLINE_SUBDIR; e o precedente usado por 229.",
    },
    {
        "component": "Diagnostico PT checkpoint205",
        "scripts": "206_diagnosticar_pt_severo_sem_oracle_checkpoint205_padding_512.py",
        "status": "reutilizavel_com_env",
        "notes": "Aceita SPINAL_206_CENTERLINE_SUBDIR.",
    },
    {
        "component": "Promocoes checkpoint",
        "scripts": "191, 194, 198, 201, 205",
        "status": "reutilizavel_se_entradas_equivalentes_existirem",
        "notes": "Materializam checkpoints; nao escolhem novas regras.",
    },
]

NEEDS_REWORK_FOR_CENTERLINE = [
    {
        "component": "MT 190 -> 191",
        "scripts": "190_validar_mt188_protegido_checkpoint186_padding_512.py",
        "required_change": "Regenerar MT_CANDIDATES_DIR para cada CENTERLINE_SUBDIR.",
        "blocker": "Le candidatos antigos de validar_selector_mt_diversidade_checkpoint114_global_top1200_v1.",
    },
    {
        "component": "PT 193 -> 194 e 200 -> 201",
        "scripts": "193_validar_pt_focado_checkpoint191_padding_512.py, 200_validar_pt_residual_checkpoint198_padding_512.py",
        "required_change": "Regenerar PT_CANDIDATES_DIR equivalente para cada CENTERLINE_SUBDIR.",
        "blocker": "193 le gate_pt_hard_negatives_checkpoint114_v1, calculado sobre artefactos antigos.",
    },
    {
        "component": "TL/L 196 -> 197 -> 198",
        "scripts": "196_validar_tll_focado_checkpoint194_padding_512.py, 197_validar_combo_tll196_checkpoint194_padding_512.py",
        "required_change": "Regenerar TLL_OLD_CANDIDATES_DIR, TLL159_DIR e TLL175_DIR para cada CENTERLINE_SUBDIR.",
        "blocker": "196 le candidatos antigos e nao troca centerline diretamente neste ponto da cadeia.",
    },
    {
        "component": "Severe-first 203 -> 204 -> 205",
        "scripts": "203_preparar_severe_first_checkpoint201_padding_512.py, 204_validar_severe_first_pt_tll_checkpoint201_padding_512.py",
        "required_change": "Regenerar candidatos severe-first PT/TL-L a partir dos candidatos PT/TL-L equivalentes.",
        "blocker": "203 depende de PT193/TLL196; sem esses candidatos regenerados por centerline, nao e fiel.",
    },
    {
        "component": "211/229",
        "scripts": "211_validar_selector_pt_features_v2.py, 229_gerar_candidatos_validar_trainfull_checkpoint205_padding_512.py",
        "required_change": "Definir se 211 e parte congelada ou diagnostico OOF antes de incluir na pipeline final.",
        "blocker": "211 revalida modelos OOF com labels diagnosticas; nao e, sozinho, um selector congelado.",
    },
]

FORBIDDEN_OLD_ARTIFACTS = [
    "validar_selector_mt_diversidade_checkpoint114_global_top1200_v1",
    "gate_pt_hard_negatives_checkpoint114_v1",
    "gate_two_stage_hard_negatives_tll_v1",
    "gerar_candidatos_tll_checkpoint156_relevantes_v1",
    "gerar_candidatos_tll_remaining_selector173_v1",
    "diagnosticar_falhas_checkpoint191_v1",
    "diagnosticar_falhas_checkpoint194_v1",
    "diagnosticar_falhas_checkpoint198_v1",
    "auditar_gargalo_checkpoint201_v1",
    "diagnostico_pt_severo_sem_oracle_checkpoint205_v1",
]


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_dir_has_results(path):
    return path.exists() and any(path.iterdir())


def prepare_output_dir():
    if output_dir_has_results(OUTPUT_DIR):
        summary_path = OUTPUT_DIR / "242_summary.md"
        if not summary_path.exists():
            raise FileExistsError(f"Pasta de output ja tem ficheiros: {OUTPUT_DIR}")
    TRAIN2000_DIR.mkdir(parents=True, exist_ok=True)
    TRAINFULL_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def count_centerlines(subdir):
    path = CENTERLINES_DIR / subdir
    if not path.exists():
        raise FileNotFoundError(f"Pasta de centerlines nao encontrada: {path}")
    files = sorted(path.glob("*_centerline.png"))
    if not files:
        raise FileNotFoundError(f"Nenhuma centerline encontrada em: {path}")
    return path, len(files)


def write_step_plan(train2000_count, trainfull_count):
    lines = [
        "# Plano 242 - reaplicar pipeline refinada final",
        "",
        "## Objetivo",
        "",
        "Comparar o efeito da centerline mantendo a pipeline Cobb fixa.",
        "",
        "## Fontes",
        "",
        f"- train2000: `{TRAIN2000_CENTERLINE_SUBDIR}` ({train2000_count} centerlines).",
        f"- trainfull: `{TRAINFULL_CENTERLINE_SUBDIR}` ({trainfull_count} centerlines).",
        "",
        "## Cadeia historica alvo",
        "",
        "- MT: 190 -> 191.",
        "- PT: 193 -> 194, 200 -> 201, e 211/229 apenas se for confirmado como parte congelada.",
        "- TL/L: 196 -> 197 -> 198.",
        "- Severe-first: 203 -> 204 -> 205.",
        "",
        "## Passos fieis necessarios",
        "",
        "1. Calcular Cobb direto base para cada centerline.",
        "2. Materializar checkpoint inicial compativel para cada fonte.",
        "3. Regenerar candidatos MT/PT/TL-L para cada fonte, com os mesmos geradores/configs historicos.",
        "4. Reaplicar 190, 193, 196/197, 200, 203/204 na mesma ordem.",
        "5. Materializar 191, 194, 198, 201 e 205.",
        "6. Comparar train2000 vs trainfull apenas no checkpoint final reaplicado.",
        "",
        "## Estado atual",
        "",
        "Bloqueado antes da execucao: nem todos os geradores historicos de candidatos estao expostos por CENTERLINE_SUBDIR.",
    ]
    (OUTPUT_DIR / "242_step_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(train2000_path, train2000_count, trainfull_path, trainfull_count):
    lines = [
        "# Manifest 242",
        "",
        "## Escopo",
        "",
        "- Reaplicar a pipeline refinada final mantendo regras/configs fixas.",
        "- Mudar apenas a fonte geometrica: centerline train2000 vs trainfull.",
        "- Nao promover checkpoints.",
        "- Nao criar selectors novos.",
        "- Nao criar guards novos.",
        "- Nao usar GT na inferencia.",
        "",
        "## Centerlines verificadas",
        "",
        f"- train2000: `{train2000_path}` ({train2000_count}).",
        f"- trainfull: `{trainfull_path}` ({trainfull_count}).",
        "",
        "## Componentes reutilizaveis diretamente",
        "",
    ]
    for item in DIRECTLY_REUSABLE:
        lines.append(f"- {item['component']}: `{item['scripts']}`; {item['status']}. {item['notes']}")
    lines.extend(["", "## Componentes que precisam de refatoracao/regeneracao", ""])
    for item in NEEDS_REWORK_FOR_CENTERLINE:
        lines.append(f"- {item['component']}: `{item['scripts']}`. {item['required_change']} Bloqueio: {item['blocker']}")
    lines.extend(["", "## Artefactos antigos proibidos nesta comparacao", ""])
    for artifact in FORBIDDEN_OLD_ARTIFACTS:
        lines.append(f"- `{artifact}`")
    lines.extend([
        "",
        "## Decisao",
        "",
        "- Bloqueado por fidelidade metodologica.",
        "- A comparacao final nao foi calculada porque reutilizar candidatos antigos violaria a regra 'so muda CENTERLINE_SUBDIR'.",
    ])
    (OUTPUT_DIR / "242_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocked_outputs():
    blocked = [{
        "status": "blocked",
        "reason": "pipeline_190_205_nao_reaplicavel_fielmente_sem_regeneradores_MT_PT_TLL_por_centerline",
    }]
    write_csv(COMPARISON_DIR / "242_train2000_metrics.csv", blocked)
    write_csv(COMPARISON_DIR / "242_trainfull_metrics.csv", blocked)
    write_csv(COMPARISON_DIR / "242_comparison_summary.csv", blocked)
    write_csv(COMPARISON_DIR / "242_regression_summary.csv", blocked)

    lines = [
        "# Resumo 242",
        "",
        "## Estado",
        "",
        "- Bloqueado antes da execucao da comparacao.",
        "",
        "## Motivo",
        "",
        "A pipeline refinada final 190-205 depende de candidatos historicos MT/PT/TL-L.",
        "Esses candidatos foram calculados sobre centerlines antigas e/ou janelas parciais.",
        "Usa-los nesta comparacao misturaria pipeline e geometria, violando a metodologia.",
        "",
        "## Proximo passo tecnico correto",
        "",
        "Criar wrappers/refatoracoes para regenerar, por CENTERLINE_SUBDIR, os candidatos equivalentes de MT, PT e TL/L usados por 190, 193, 196/197, 200 e 203/204.",
        "So depois o 242 deve calcular as metricas finais train2000 vs trainfull.",
    ]
    (OUTPUT_DIR / "242_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    prepare_output_dir()
    train2000_path, train2000_count = count_centerlines(TRAIN2000_CENTERLINE_SUBDIR)
    trainfull_path, trainfull_count = count_centerlines(TRAINFULL_CENTERLINE_SUBDIR)

    write_step_plan(train2000_count, trainfull_count)
    write_manifest(train2000_path, train2000_count, trainfull_path, trainfull_count)
    write_blocked_outputs()

    raise RuntimeError(
        "242 bloqueado por fidelidade: faltam regeneradores equivalentes MT/PT/TL-L por CENTERLINE_SUBDIR. "
        f"Ver {OUTPUT_DIR / '242_summary.md'}"
    )


if __name__ == "__main__":
    main()
