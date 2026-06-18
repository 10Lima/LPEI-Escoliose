import os
import runpy
from pathlib import Path


# =========================
# CONFIGURACOES
# =========================

# Este script continua a ser a etapa oficial 06 da pipeline.
# A implementacao reutiliza o gerador parametrizavel do script 11.
DEFAULT_NUM_EXAMPLES = "100"
DEFAULT_OUTPUT_SUBDIR = "unet_baseline_2000_padding_512_centerline_100"


def main():
    os.environ.setdefault("SPINAL_CENTERLINE_NUM_EXAMPLES", DEFAULT_NUM_EXAMPLES)
    os.environ.setdefault("SPINAL_CENTERLINE_OUTPUT_SUBDIR", DEFAULT_OUTPUT_SUBDIR)

    script_path = (
        Path(__file__).resolve().parent
        / "11_gerar_centerlines_validacao_500_padding_512.py"
    )

    if not script_path.exists():
        raise FileNotFoundError(f"Script 11 nao encontrado: {script_path}")

    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
