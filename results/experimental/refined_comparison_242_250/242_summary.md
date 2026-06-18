# Resumo 242

## Estado

- Bloqueado antes da execucao da comparacao.

## Motivo

A pipeline refinada final 190-205 depende de candidatos historicos MT/PT/TL-L.
Esses candidatos foram calculados sobre centerlines antigas e/ou janelas parciais.
Usa-los nesta comparacao misturaria pipeline e geometria, violando a metodologia.

## Proximo passo tecnico correto

Criar wrappers/refatoracoes para regenerar, por CENTERLINE_SUBDIR, os candidatos equivalentes de MT, PT e TL/L usados por 190, 193, 196/197, 200 e 203/204.
So depois o 242 deve calcular as metricas finais train2000 vs trainfull.
