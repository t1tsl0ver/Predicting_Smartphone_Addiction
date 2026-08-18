# trial-040 — Weekend screen-time value frequency

- Статус: **rejected; weak positive point estimate, mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

Изолированно добавлена fold-safe частота точного `weekend_screen_time`; target
не используется, остальной протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0000448023 |
| Base LightGBM | +0.0004133717 |
| 60/40 blend | **+0.0000286671** |

Blend OOF: `0.9645386677 → 0.9645673348`. Fold-deltas:
`+0.0000656204` и `-0.0000079464`. Paired interval:
`[-0.0000345, +0.0000852]`, положительных повторов `81/100`.

## Вывод

`weekend_screen_time_value_frequency` **отклонена**: base улучшился, но exact
ухудшился, один fold отрицателен, paired interval пересекает ноль.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-040 --candidate weekend_screen_time_value_frequency
```

Артефакты: `artifacts/metrics_trial-040_feature.json`, OOF, importance и лог
`20260818T125750376171+0300_cv-feature-trial_trial-040.log`.
