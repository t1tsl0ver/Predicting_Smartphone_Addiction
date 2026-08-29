# trial-066 — Ablation numeric value mean surprisal

- Статус: **accepted as a simplification; AUC change indistinguishable from zero**
- Дата: 2026-08-29
- Контроль: `trial-065`, OOF `0.9666455685`

## Вопрос и гипотеза

Нужен ли `numeric_value_mean_surprisal` в текущем восьмифичевом baseline?
`trial-051` добавил его одним пакетом с четырьмя screening-победителями, поэтому
его самостоятельный вклад не был установлен. Если удаление не ухудшает OOF,
предпочтителен более простой и лучше объяснимый пакет.

## Единственное изменение

Из `trial-065` удалён только:

```text
numeric_value_mean_surprisal
```

Остались четыре фичи `trial-043`, `social_screen_share`, `other_screen_share` и
`gaming_screen_share`. Неизменны 5 folds, seed `2026`, LightGBM 200 деревьев,
LR `0.10` и exact/base fold-rank blend `60/40`.

## Результаты

| Компонент | `trial-065` | Без surprisal | Delta |
|---|---:|---:|---:|
| Exact | 0.9660670799 | **0.9660775685** | +0.0000104886 |
| Base | 0.9623557041 | **0.9623802618** | +0.0000245577 |
| Blend | 0.9666455685 | **0.9666541546** | **+0.0000085861** |

Fold-delta blend: `-0.0000643481`, `+0.0000208721`, `-0.0000473117`,
`+0.0000383504`, `+0.0000983814`: положительны 3/5 folds.

Paired bootstrap на 300 повторах дал CI95
`[-0.0000300903; +0.0000486598]`, `P(delta>0)=0.67`. Интервал пересекает ноль;
статистически подтверждённого улучшения нет. Spearman predictions равен
`0.9987637`, то есть варианты почти эквивалентны по ранжированию.

## Вывод

Удаление **не является доказанным улучшением качества**, но и измеримого вреда
не показало: точечная оценка немного выше, а неопределённость мала относительно
текущего разрыва до `exp-005`. По заранее согласованному правилу «при равенстве
предпочесть более простой пакет» surprisal удалён.

Семифичевый кандидат становится новым лёгким baseline с OOF `0.9666541546`.
До no-TE ориентира `exp-005` `0.9668743680` осталось `0.0002202134`. Submission
не создавался.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-066 --candidate trial_065_without_numeric_value_mean_surprisal --control-oof artifacts\oof_predictions_trial-065.csv --control-metrics artifacts\metrics_trial-065_feature.json --folds 5 --iterations 200 --bootstrap-repeats 300 --log-every 100
```

Артефакты: `artifacts/metrics_trial-066_feature.json`, OOF, importance и лог
`20260829T200015112865+0300_cv-feature-trial_trial-066.log`.
