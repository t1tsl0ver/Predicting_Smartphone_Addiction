# trial-034 — Social-media value frequency

- Статус: **rejected; negative paired interval and lost 2/2 folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Изменение

Изолированно добавлена fold-safe частота точного `social_media_hours`. Mapping
строится только на train-части fold, unseen validation получает `0`, пропуск —
отдельное значение, target не используется. Остальной протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0001007362 |
| Base LightGBM | -0.0007318693 |
| 60/40 blend | **-0.0001503799** |

Blend OOF: `0.9645386677 → 0.9643882878`. Fold-deltas:
`-0.0000718050` и `-0.0002289159`.

Paired bootstrap на 100 повторах размера test дал полностью отрицательный
interval `[-0.0002187, -0.0000681]`; положительных повторов `0/100`.

## Вывод

`social_media_hours_value_frequency` **уверенно отклонена**: общий результат,
оба folds, оба компонента и paired interval отрицательны.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-034 --candidate social_media_hours_value_frequency
```

Артефакты: `artifacts/metrics_trial-034_feature.json`, OOF, importance и лог
`20260818T124511434112+0300_cv-feature-trial_trial-034.log`.
