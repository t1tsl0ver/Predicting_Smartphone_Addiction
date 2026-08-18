# trial-041 — Missing-pattern frequency

- Статус: **rejected; effectively zero delta and mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Изменение

Для 12 исходных признаков построена стабильная 12-битная маска пропусков. В оба
компонента добавлена одна числовая fold-safe частота полной маски:

```text
missing_pattern_frequency = count(full missing bitmask) / fold train rows
```

Mapping строится только по train-части fold, unseen validation получает `0`,
target не используется. `missing_feature_count` из `trial-031` не добавляется.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | +0.0000067109 |
| Base LightGBM | -0.0002409878 |
| 60/40 blend | **+0.0000017745** |

Blend OOF: `0.9645386677 → 0.9645404422`. Fold-deltas:
`+0.0000233867` и `-0.0000198564`. Paired interval:
`[-0.0000315, +0.0000370]`, положительных повторов ровно `50/100`.

## Вывод

`missing_pattern_frequency` **отклонена**: общая дельта практически нулевая,
folds расходятся по знаку, bootstrap симметрично пересекает ноль. Это не
исключает отдельную проверку самой mask identity как categorical-фичи.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-041 --candidate missing_pattern_frequency
```

Артефакты: `artifacts/metrics_trial-041_feature.json`, OOF, importance и лог
`20260818T130044131367+0300_cv-feature-trial_trial-041.log`.
