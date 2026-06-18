# Resumo 247

## Estado

- Severe-first ainda nao esta valido para alimentar o 248.

## Cobertura

- train2000: PT 2248/3192 (1549122 candidatos, oracle <=5 34.9937%, <=10 59.2105%); TL/L 3175/3192 (3807930 candidatos, oracle <=5 71.4912%, <=10 83.584%).
- trainfull: PT 2256/3192 (1557752 candidatos, oracle <=5 35.2757%, <=10 59.2732%); TL/L 3177/3192 (3810372 candidatos, oracle <=5 71.8358%, <=10 83.4273%).

## Nota metodologica

- Este script nao desbloqueia o 242 sozinho.
- O proximo passo e o 248, que deve decidir se estes artefactos sao suficientes para reaplicar a pipeline fixa sem usar GT na inferencia.
