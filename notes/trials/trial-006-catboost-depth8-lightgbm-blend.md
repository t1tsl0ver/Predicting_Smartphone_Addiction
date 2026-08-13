# trial-006 — CatBoost d8/1000 + LightGBM rank blend

- Статус: **weak positive; CV required**
- Дата: 2026-08-13
- Контроль: LightGBM baseline из `trial-003`

## Вопрос

Может ли лучший проверенный CatBoost d8/1000 улучшить LightGBM в rank blend после
того, как d7/1000 практически сравнялся с ним, но не превзошёл?

## Компоненты

| Модель | Local ROC AUC |
|---|---:|
| CatBoost d8/1000 | 0.9613782447 |
| LightGBM | **0.9628707172** |

Обе модели были обучены ранее на одинаковом split. В этом trial выполнялось
только смешивание сохранённых validation predictions.

## Разнообразие прогнозов

```text
Pearson correlation:  0.992966
Spearman correlation: 0.990097
```

Depth 8 сильнее depth 7, но его ранги ещё немного ближе к LightGBM, чем ранги
d7/1000 (`0.990097` против `0.989218`).

## Rank blends

| CatBoost | LightGBM | Validation AUC | Δ к LightGBM |
|---:|---:|---:|---:|
| 15% | 85% | **0.9629179814** | **+0.0000472642** |
| 30% | 70% | 0.9628698043 | -0.0000009129 |
| 50% | 50% | 0.9626563234 | -0.0002143938 |
| 70% | 30% | 0.9622729016 | -0.0005978156 |
| 85% | 15% | 0.9618735385 | -0.0009971787 |

## Вывод

Вес 15/85 формально дал новый лучший holdout AUC, но прирост `+0.0000473` меньше
заранее установленного уровня `+0.0001`, который считался практически
незначимым. После нескольких сравнений на одном holdout такая разница легко может
быть следствием адаптации к validation.

Ансамбль сохраняется как слабый кандидат для OOF/CV-проверки, но пока не считается
надёжно принятым улучшением. Подбирать ещё более точные веса на том же holdout не
следует.

## Воспроизведение

```powershell
.\.venv\Scripts\python.exe src\blend.py --primary artifacts\validation_predictions_trial-004_catboost.csv --secondary artifacts\validation_predictions_trial-003_lightgbm.csv --primary-name catboost-d8-1000 --secondary-name lightgbm --run-id trial-006
```
