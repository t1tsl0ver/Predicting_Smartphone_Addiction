# trial-064 — Доля gaming в общем экранном времени

- Статус: **accepted screening**
- Дата: 2026-08-22
- Контроль: `trial-054`, OOF `0.9662173704`

## Вопрос и гипотеза

Поможет ли модели явная доля gaming, хотя она алгебраически выражается через
уже имеющиеся entertainment- и social-доли? Деревьям может быть проще разбивать
готовую координату, чем восстанавливать разность двух отношений.

## Единственное изменение

К шести фичам `trial-054` добавлена только
`gaming_screen_share = gaming_hours / daily_screen_time_hours`. Нулевой
знаменатель и пропуски дают `NaN`. Target и статистики датасета не используются.
Неизменны 5 folds, seed `2026`, LightGBM 200 деревьев, LR `0.10` и exact/base
fold-rank blend `60/40`.

## Результаты

Exact delta `+0.0003258077`, base `+0.0001140909`, blend
**`+0.0002230501`** до **`0.9664404205`**. Fold-delta:
`+0.0002127889`, `+0.0002091657`, `+0.0002419782`, `+0.0001904843`,
`+0.0002557544`: победа 5/5.

Paired bootstrap 100 повторов: CI95
`[+0.0001238126; +0.0003138712]`, `P(delta>0)=1.00`. Время: `299.6 s`.
Mean gain фичи: `24562.4` в exact и `68113.1` в base.

## Вывод

Фича устойчиво усилила обе компоненты и принята для совместной проверки с
`trial-063 other_screen_share`.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-064 --candidate gaming_screen_share --include-trial-054-baseline --control-oof artifacts\oof_predictions_trial-054.csv --control-metrics artifacts\metrics_trial-054_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

