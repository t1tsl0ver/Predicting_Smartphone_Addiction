# trial-057 — Концентрация состава экранного времени

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Даёт ли более простая, чем entropy, мера концентрации экранного бюджета сигнал
генератора? Фича — сумма квадратов долей social, gaming, work/study и `other`.

## Изменение и контроль

Добавлена только `screen_mix_concentration`; обработка невозможных разложений и
вся 5-fold/200-tree конфигурация совпадают с `trial-056` и контролем.

## Результаты

Exact delta `+0.0000071383`, base `-0.0000216422`, blend
**`+0.0000056027`** до `0.9660283688`. Fold-delta: `+0.0000579729`,
`-0.0000364537`, `+0.0000270308`, `-0.0000365274`, `+0.0000126104` — 3/5.
Bootstrap CI95 `[-0.0000396299; +0.0000482511]`, `P(delta>0)=0.62`, 100
повторов. Время: `217.2 s`.

## Вывод

Результат практически нулевой и нестабильный. Фича отклонена.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-057 --candidate screen_mix_concentration --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

