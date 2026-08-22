# trial-063 — Доля прочего экранного времени

- Статус: **accepted screening**
- Дата: 2026-08-22
- Контроль: `trial-054`, OOF `0.9662173704`

## Вопрос и гипотеза

Абсолютный residual `other_screen_time_hours` уже полезен. Добавит ли модельный
сигнал его относительная величина — доля не объяснённого social, gaming и
work/study времени в общем daily screen time?

## Единственное изменение

К шести фичам `trial-054` добавлена только
`other_screen_share = (daily - social - gaming - work) / daily`. Нулевой
знаменатель и любой необходимый пропуск дают `NaN`. Target и статистики
датасета не используются. Неизменны 5 folds, seed `2026`, LightGBM 200
деревьев, LR `0.10` и exact/base fold-rank blend `60/40`.

## Результаты

Exact delta `+0.0002708863`, base `+0.0001846265`, blend
**`+0.0002073118`** до **`0.9664246822`**. Fold-delta:
`+0.0002606184`, `+0.0001727515`, `+0.0002536333`, `+0.0002299452`,
`+0.0001155406`: победа 5/5.

Paired bootstrap 100 повторов: CI95
`[+0.0001369415; +0.0002976677]`, `P(delta>0)=1.00`. Время: `204.5 s`.
Mean gain фичи: `16917.5` в exact и `28663.9` в base.

## Вывод

Фича устойчиво усилила обе компоненты и принята для последующей комбинации.
Это не TE и не fingerprint lookup: формула использует только значения строки.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-063 --candidate other_screen_share --include-trial-054-baseline --control-oof artifacts\oof_predictions_trial-054.csv --control-metrics artifacts\metrics_trial-054_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

