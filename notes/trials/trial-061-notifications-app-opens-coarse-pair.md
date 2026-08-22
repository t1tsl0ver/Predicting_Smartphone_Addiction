# trial-061 — Фиксированная пара notifications × app opens

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Есть ли у генератора совместные режимы количества уведомлений и открытий
приложений? Колонки разделены фиксированными границами `50/100/150/200` и
`40/80/120/160`; категории объединены без target или frequency lookup.

## Изменение и контроль

Добавлена только `notifications_app_opens_coarse_pair`; неизменны 5 folds, seed
`2026`, LightGBM 200 деревьев, LR `0.10` и blend `60/40`.

## Результаты

Exact delta `-0.0000141472`, base `-0.0002692288`, blend
**`-0.0000671579`** до `0.9659556081`. Fold-delta: `-0.0000820688`,
`-0.0000557341`, `-0.0000912208`, `-0.0000909095`, `-0.0000169348` —
проигрыш 5/5. Bootstrap CI95 `[-0.0001090427; -0.0000214421]`,
`P(delta>0)=0.00`, 100 повторов. Время: `200.9 s`.

## Вывод

Пара стабильно ухудшает blend и особенно base-компонент. Фича отклонена.
Серия завершена; единственным строгим победителем остался `trial-054`.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-061 --candidate notifications_app_opens_coarse_pair --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```
