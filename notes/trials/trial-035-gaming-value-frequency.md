# trial-035 — Gaming value frequency

- Статус: **rejected; negative point estimate and mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Изменение

Изолированно добавлена fold-safe частота точного `gaming_hours`; mapping строится
только на train-части fold, target не используется. Остальной протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0000652832 |
| Base LightGBM | -0.0003162742 |
| 60/40 blend | **-0.0000123591** |

Blend OOF: `0.9645386677 → 0.9645263086`. Fold-deltas:
`+0.0000098857` и `-0.0000338856`. Paired bootstrap interval:
`[-0.0000500, +0.0000385]`, положительных повторов `37/100`.

## Вывод

`gaming_hours_value_frequency` **отклонена**: общая дельта и оба компонента
отрицательны, folds расходятся по знаку, интервал пересекает ноль.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-035 --candidate gaming_hours_value_frequency
```

Артефакты: `artifacts/metrics_trial-035_feature.json`, OOF, importance и лог
`20260818T124723352698+0300_cv-feature-trial_trial-035.log`.
