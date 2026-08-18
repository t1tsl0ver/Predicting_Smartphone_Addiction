# trial-036 — Work/study value frequency

- Статус: **rejected; weak positive point estimate, mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Изменение

Изолированно добавлена fold-safe частота точного `work_study_hours`. Mapping
строится только по train fold, target не используется; остальные параметры
неизменны. Победившая `work_study_screen_share` из `trial-029` не добавляется.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | +0.0000099720 |
| Base LightGBM | -0.0001877098 |
| 60/40 blend | **+0.0000157682** |

Blend OOF: `0.9645386677 → 0.9645544359`. Fold-deltas:
`+0.0000402900` и `-0.0000085587`. Paired interval:
`[-0.0000373, +0.0000695]`, положительных повторов `64/100`.

## Вывод

`work_study_hours_value_frequency` **отклонена**: положительный point estimate
мал, один fold отрицателен, paired interval пересекает ноль. Результат не меняет
статус сильной отдельной доли work/study из `trial-029`.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-036 --candidate work_study_hours_value_frequency
```

Артефакты: `artifacts/metrics_trial-036_feature.json`, OOF, importance и лог
`20260818T124933536914+0300_cv-feature-trial_trial-036.log`.
