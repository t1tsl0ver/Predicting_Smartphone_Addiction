# trial-053 — Доля social внутри entertainment

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Есть ли у генератора отдельный сигнал в составе развлечений, если разделить
social и gaming независимо от общего экранного времени? Проверяем
`social_media_hours / (social_media_hours + gaming_hours)` без target lookup.

## Изменение и контроль

К пяти фичам `trial-051` добавлена только `social_within_entertainment`. Нулевой
знаменатель и пропуски дают `NaN`. Неизменны 5 stratified folds, seed `2026`,
LightGBM 200 деревьев, LR `0.10` и exact/base fold-rank blend `60/40`.

## Результаты

| Компонент | OOF delta |
|---|---:|
| Exact | +0.0000986624 |
| Base | -0.0000713104 |
| Blend | **+0.0000147826** до `0.9660375486` |

Fold-delta blend: `+0.0001131010`, `-0.0000259229`, `-0.0000593232`,
`-0.0000986516`, `+0.0001424615`. Положительны 2/5 folds. Paired bootstrap
100 повторов: CI95 `[-0.0000741978; +0.0001048184]`, `P(delta>0)=0.67`.
Время: `180.5 s`.

## Вывод

Точечный плюс слишком мал, base ухудшился, folds смешанные, interval пересекает
ноль. Фича отклонена и не входит в объединение победителей.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-053 --candidate social_within_entertainment --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

