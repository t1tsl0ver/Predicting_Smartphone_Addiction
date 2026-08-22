# trial-060 — Фиксированная пара social-share × work-share

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Есть ли у генератора совместные режимы социальных и рабочих долей экрана?
Каждая доля разделена фиксированными границами `0.2/0.4/0.6/0.8`, после чего
две корзины объединены в categorical-пару без target или frequency lookup.

## Изменение и контроль

Добавлена только `social_work_share_coarse_pair`; неизменны 5 folds, seed
`2026`, LightGBM 200 деревьев, LR `0.10` и blend `60/40`.

## Результаты

Exact delta `-0.0001437323`, base `-0.0002397007`, blend
**`-0.0001462584`** до `0.9658765077`. Fold-delta: `+0.0000127011`,
`-0.0002261078`, `-0.0001919079`, `-0.0001317695`, `-0.0001963227` — 1/5.
Bootstrap CI95 `[-0.0002417157; -0.0000711324]`, `P(delta>0)=0.00`, 100
повторов. Время: `198.3 s`.

## Вывод

Грубая совместная категория хуже непрерывных долей и уверенно отклонена.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-060 --candidate social_work_share_coarse_pair --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

