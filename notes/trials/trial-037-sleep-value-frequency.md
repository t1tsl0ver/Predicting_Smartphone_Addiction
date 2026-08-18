# trial-037 — Sleep value frequency

- Статус: **rejected; negative interval and lost 2/2 folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

Изолированно добавлена fold-safe частота точного `sleep_hours`; target не
используется, остальной протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0000368262 |
| Base LightGBM | -0.0003247861 |
| 60/40 blend | **-0.0000405637** |

Blend OOF: `0.9645386677 → 0.9644981041`. Fold-deltas:
`-0.0000673141` и `-0.0000136472`. Paired interval:
`[-0.0000724, -0.0000039]`; положителен 1/100 повторов.

## Вывод

`sleep_hours_value_frequency` **отклонена**: оба folds, оба компонента, общий
OOF и paired interval отрицательны.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-037 --candidate sleep_hours_value_frequency
```

Артефакты: `artifacts/metrics_trial-037_feature.json`, OOF, importance и лог
`20260818T125136810788+0300_cv-feature-trial_trial-037.log`.
