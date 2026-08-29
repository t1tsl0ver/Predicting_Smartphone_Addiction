# exp-010 — Rank blend of exp-009 and exp-007

- Статус: **completed; current local and leaderboard leader**
- Дата: 2026-08-29
- Компоненты: `exp-009` no-TE и исторический `exp-007` leak-safe TE
- Метод: 55/45 weighted rank blend без переобучения

## Цель

Проверить, может ли новый сильный no-TE `exp-009` заменить старую no-TE часть
ансамбля `exp-008`, сохранив полезное разнообразие `exp-007`. Использованы уже
сохранённые OOF/test predictions; новые модели не обучались.

## Sweep весов

Перебрана доля `exp-009` от 0% до 100% с шагом 1%.

| Вес `exp-009` | Вес `exp-007` | OOF AUC |
|---:|---:|---:|
| 50% | 50% | 0.9675180870 |
| 55% | 45% | **0.9675281811** |
| 57% | 43% | **0.9675290840** |
| 60% | 40% | 0.9675274433 |
| 65% | 35% | 0.9675160975 |

Формальный максимум находится на 57/43. Диапазон 53–62% по `exp-009`
уступает ему меньше `0.000005`. Выбран простой вес 55/45 в центре плато; он
ниже формального максимума лишь на `0.0000009029`.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\blend_cv.py --primary-oof artifacts\oof_predictions_exp-009.csv --primary-test artifacts\test_predictions_exp-009.csv --primary-oof-column lightgbm_fold_rank --primary-test-column lightgbm_cv_rank --primary-name exp-009 --primary-weight 0.55 --secondary-oof artifacts\oof_predictions_exp-007.csv --secondary-test artifacts\test_predictions_exp-007.csv --secondary-oof-column lightgbm_fold_rank --secondary-test-column lightgbm_cv_rank --secondary-name exp-007 --run-id exp-010 --bootstrap-repeats 300 --bootstrap-sample-size 296302 --bootstrap-seed 2026 --submission submission.csv
```

## Локальный результат

| Модель | OOF ROC AUC | Delta к `exp-008` |
|---|---:|---:|
| `exp-009` | 0.9671371213 | -0.0002013897 |
| `exp-008` | 0.9673385110 | — |
| `exp-010` 55/45 blend | **0.9675281811** | **+0.0001896701** |

| Fold | `exp-008` | `exp-010` | Delta |
|---:|---:|---:|---:|
| 1 | 0.9665431218 | **0.9667705095** | +0.0002273877 |
| 2 | 0.9677966116 | **0.9680140362** | +0.0002174246 |
| 3 | 0.9682247352 | **0.9683959373** | +0.0001712021 |
| 4 | 0.9676062913 | **0.9677251360** | +0.0001188447 |
| 5 | 0.9665210266 | **0.9667329972** | +0.0002119706 |

Прямой paired bootstrap против `exp-008` на 300 выборках размера test дал mean
delta `+0.0001893771`, 95% interval `[+0.0001349506, +0.0002505363]` и
`300/300` положительных повторов. Интервал учитывает row-sampling uncertainty,
но не подбор веса и leaderboard split.

## Submission

```text
file=submission.csv
rows=296302
prediction=0.55 * exp-009 rank + 0.45 * exp-007 rank
min=0.000008984077
max=0.999893554549
sha256=ebfd43a202f93b25918d4f07c9e257eae6ed02c4f71a2575de451d0806b1e905
Kaggle score=0.96874
```

Схема, порядок ID, конечность predictions и соответствие SHA-256 metadata
проверены. Kaggle score вырос с `0.96863` у `exp-008` до `0.96874`: фактический
leaderboard-прирост **`+0.00011`**. Направление OOF подтвердилось, хотя прирост
на leaderboard оказался меньше локальной дельты `+0.0001897`.

## Вывод

`exp-010` — новый локальный и leaderboard-лидер. Усиление no-TE компонента
сохранило комплементарность исторического `exp-007`, устойчиво превысило
`exp-008` на всех folds и подтвердилось на Kaggle.
