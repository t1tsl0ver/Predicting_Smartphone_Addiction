# trial-059 — Фиксированная пара daily × weekend

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Есть ли у генератора дискретные совместные режимы daily и weekend screen time,
которые одиночным splits восстановить сложнее? Обе величины разбиты заранее
фиксированными границами (`3/6/9/12` и `4/8/12/16`), затем объединены в одну
categorical-пару. Target и частоты не используются.

## Изменение и контроль

Добавлена только `daily_weekend_coarse_pair`; неизменны 5 folds, seed `2026`,
LightGBM 200 деревьев, LR `0.10` и blend `60/40`.

## Результаты

Exact delta `-0.0000542124`, base `-0.0002747496`, blend
**`-0.0000707283`** до `0.9659520378`. Fold-delta: `-0.0000125620`,
`-0.0001426985`, `+0.0000168508`, `-0.0002266160`, `+0.0000068940` — 2/5.
Bootstrap CI95 `[-0.0001493632; -0.0000034681]`, `P(delta>0)=0.01`, 100
повторов. Время: `211.8 s`.

## Вывод

Пара ухудшает обе компоненты, а interval полностью отрицателен. Отклонена.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-059 --candidate daily_weekend_coarse_pair --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

