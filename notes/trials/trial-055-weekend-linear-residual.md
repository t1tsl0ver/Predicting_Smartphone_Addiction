# trial-055 — Fold-safe линейный weekend residual

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Есть ли сигнал в отклонении weekend screen time от типичной зависимости с daily
screen time? На каждом fold обычная OLS `weekend ~ daily` обучается только на
fold-train, затем сохраняется residual. Target не используется.

## Изменение и контроль

К пяти фичам `trial-051` добавлен только `weekend_linear_residual`. Неизменны 5
folds, seed `2026`, LightGBM 200 деревьев, LR `0.10`, blend `60/40`.

## Результаты

Exact delta `+0.0000335483`, base `-0.0000276664`, blend
**`+0.0000214142`** до `0.9660441802`. Fold-delta: `+0.0000180173`,
`+0.0000062506`, `-0.0000083573`, `-0.0000377018`, `+0.0001278384` — 3/5.
Bootstrap CI95 `[-0.0000448515; +0.0000801219]`, `P(delta>0)=0.76`, 100
повторов. Время: `201.8 s`.

## Вывод

Слабый точечный плюс неустойчив и не компенсирует ухудшение base-компонента.
Фича отклонена и не входит в объединение.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-055 --candidate weekend_linear_residual --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

