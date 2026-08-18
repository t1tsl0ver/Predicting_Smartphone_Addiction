# trial-039 — App-opens value frequency

- Статус: **rejected; weak positive point estimate, one fold non-positive**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

Изолированно добавлена fold-safe частота точного `app_opens_per_day`; target не
используется, остальной протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0000264694 |
| Base LightGBM | +0.0002766837 |
| 60/40 blend | **+0.0000295699** |

Blend OOF: `0.9645386677 → 0.9645682377`. Fold-deltas:
`+0.0000602814` и `-0.0000007457`. Paired interval:
`[-0.0000287, +0.0000908]`, положительных повторов `83/100`.

## Вывод

`app_opens_per_day_value_frequency` **отклонена**: point estimate положителен,
но второй fold слегка отрицателен и paired interval пересекает ноль. Base-сигнал
интересен, однако заметно слабее и менее устойчив, чем frequency уведомлений из
`trial-038`.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-039 --candidate app_opens_per_day_value_frequency
```

Артефакты: `artifacts/metrics_trial-039_feature.json`, OOF, importance и лог
`20260818T125546599926+0300_cv-feature-trial_trial-039.log`.
