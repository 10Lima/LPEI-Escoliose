# Resumo 243

## Estado

- Bloqueado antes da geracao final de candidatos.

## Porque bloqueou

Os candidatos historicos usados pela cadeia refinada final nao estao todos expostos como regeneradores all-cases por CENTERLINE_SUBDIR.
Criar candidatos genericos agora repetiria o erro metodologico do 240/241.

## Outputs escritos

- `train2000/candidates/pt_candidates.csv`
- `train2000/candidates/mt_candidates.csv`
- `train2000/candidates/tll_candidates.csv`
- `train2000/candidates/severe_candidates.csv`
- `trainfull/candidates/pt_candidates.csv`
- `trainfull/candidates/mt_candidates.csv`
- `trainfull/candidates/tll_candidates.csv`
- `trainfull/candidates/severe_candidates.csv`
- `243_coverage_summary.csv`
- `243_manifest.md`
- `243_summary.md`

## Proximo passo correto

Refatorar primeiro os geradores historicos de candidatos para aceitarem CENTERLINE_SUBDIR e cobrirem 3192 imagens, mantendo configs antigas.
