# trial-031 — Missing feature count

- Статус: **rejected; weak positive point estimate, mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Вопрос и изменение

Несёт ли общее число отсутствующих исходных признаков отдельный сигнал о строке,
который неудобно восстановить из независимых missing-ветвей деревьев?

В оба LightGBM-компонента добавлена одна числовая фича:

```text
missing_feature_count = number of NaN among 12 original predictors
```

Feature engineering колонки в count не входят. Остальная 2-fold конфигурация и
cached контроль неизменны; submission не создаётся.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | +0.0000353174 |
| Base LightGBM | -0.0000059020 |
| 60/40 blend | **+0.0000284376** |

Blend OOF: `0.9645386677 → 0.9645671054`. Fold-deltas равны
`+0.0000756566` и `-0.0000185783`.

Paired bootstrap на 100 повторах размера test:

```text
95% interval:      [-0.0000110, +0.0000701]
positive repeats:  87/100
```

Spearman control/candidate blend равен `0.998484`. Mean gain count-фичи невелик:
`65.01` в exact и `174.27` в base.

## Вывод

`missing_feature_count` **отклонена как одиночный кандидат**: точечная дельта
положительна, но один fold проигран и paired interval пересекает ноль. В будущем
рейтинге она отмечается как слабый неподтверждённый сигнал, а не как победитель.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-031 --candidate missing_feature_count
```

Артефакты: `artifacts/metrics_trial-031_feature.json`, OOF, importance и лог
`20260818T123658284146+0300_cv-feature-trial_trial-031.log`.
