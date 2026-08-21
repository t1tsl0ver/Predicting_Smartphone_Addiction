# trial-051 — Пятифолдовая проверка пакета из пяти фич

- Статус: **accepted as the lightweight 5-fold baseline**
- Дата: 2026-08-21
- Контроль: заново обученный LightGBM exact/base 60/40 blend без четырёх
  screening-фич и без numeric surprisal

## Вопрос

Сохраняется ли польза четырёх обычных победителей `trial-043` в более надёжном
5-fold режиме, и помогает ли вместе с ними строковая оценка необычности точных
числовых значений?

## Гипотеза

Синтетический генератор может неравномерно использовать точные значения. Для
каждой строки можно оценить редкость каждого из девяти числовых значений по
частотам только в train-части fold и усреднить их surprisal. Такая фича не
использует target и потенциально отделяет типичные для генератора комбинации от
редких. Одновременно 5 folds должны надёжнее подтвердить четыре ранее принятые
screening-фичи, чем их исходная 2-fold проверка.

## Изменение

В exact- и base-компоненты одним заранее согласованным пакетом добавлены:

```text
work_study_screen_share
entertainment_screen_share
notifications_per_day_value_frequency
daily_screen_time_hours_value_frequency
numeric_value_mean_surprisal
```

`numeric_value_mean_surprisal` — среднее `-log(smoothed frequency)` по доступным
значениям девяти исходных числовых колонок. Mapping строится только по train-
части текущего fold; применяется add-one smoothing с отдельным unseen-
состоянием. Пропуски не участвуют в среднем, target не используется.

Это намеренно групповая проверка по решению участника. Она оценивает весь пакет
из пяти фич и не изолирует вклад `numeric_value_mean_surprisal`.

## Что осталось неизменным

- 5 stratified folds, shuffle, seed `2026`;
- 200 деревьев, learning rate `0.10`, без early stopping;
- exact/base fold-rank blend `60/40`;
- одинаковые folds и параметры у контроля и кандидата;
- без target encoding, decimal digits и создания submission.

Для воспроизводимого контроля добавлен `src/cv_lightweight_control.py`. Контроль
и кандидат обучались последовательно; полный модельный запуск занял около
`1:33 + 2:38`, ещё `0:54` занял bootstrap кандидата.

## Результаты

| Вариант | Control OOF | Candidate OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.9642942152 | **0.9651207118** | **+0.0008264966** |
| Base LightGBM | 0.9601132668 | **0.9618772645** | **+0.0017639977** |
| 60/40 blend | 0.9653491591 | **0.9660227664** | **+0.0006736073** |

Пакет выиграл все пять folds:

| Fold | Control blend | Candidate blend | Delta |
|---:|---:|---:|---:|
| 1 | 0.9645457093 | **0.9652246592** | **+0.0006789500** |
| 2 | 0.9659532993 | **0.9665859640** | **+0.0006326648** |
| 3 | 0.9661413833 | **0.9668549407** | **+0.0007135574** |
| 4 | 0.9656263513 | **0.9662937962** | **+0.0006674449** |
| 5 | 0.9644714204 | **0.9651527383** | **+0.0006813179** |

Разброс fold-delta узкий: от `+0.0006327` до `+0.0007136`. Paired stratified
bootstrap на 300 повторах размера test полностью положителен:
`[+0.0005733; +0.0007765]`, положительны `300/300` повторов. Интервал измеряет
только неопределённость повторной выборки строк, а не переобучения или выбора
folds.

## Importance и интерпретация

В base-компоненте среди новых фич наибольший mean gain получили frequency
уведомлений (`85569.8`) и доля work/study (`52649.6`). У
`numeric_value_mean_surprisal` gain заметно ниже: `2821.6` в base и `83.9` в
exact. Модель использовала новую фичу, но gain не доказывает положительный
причинный вклад.

Поскольку четыре старые фичи и surprisal проверялись вместе, результат нельзя
раскладывать на пять независимых эффектов. Надёжно установлено только то, что
полный пакет лучше чистого лёгкого контроля.

## Решение

Пакет из пяти фич **принят как текущий лёгкий 5-fold baseline**. Его OOF
`0.9660227664`. Целевой ориентир тяжёлого no-TE `exp-005` держим только как число
`0.9668743680`; до него лёгкой модели осталось `0.0008516016` OOF. Смешивание с
тяжёлыми моделями пока не проверяем.

Следующие лёгкие идеи сравниваем на тех же 5 folds, 200 деревьях и seed `2026`.
При этом не заявляем, что surprisal полезен отдельно, пока не будет специальной
ablation.

## Команды

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightweight_control.py --run-id trial-051-control --folds 5 --iterations 200 --log-every 50
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-051 --candidate screening_winners_plus_numeric_value_mean_surprisal --control-oof artifacts\oof_predictions_trial-051-control.csv --control-metrics artifacts\metrics_trial-051-control_lightweight_control.json --folds 5 --iterations 200 --log-every 50 --bootstrap-repeats 300
```

Артефакты: `artifacts/metrics_trial-051-control_lightweight_control.json`,
`artifacts/oof_predictions_trial-051-control.csv`,
`artifacts/metrics_trial-051_feature.json`,
`artifacts/oof_predictions_trial-051.csv`, importance-файлы и логи
`20260821T220832473044+0300_cv-lightweight-control_trial-051-control.log` и
`20260821T221019640866+0300_cv-feature-trial_trial-051.log`.
