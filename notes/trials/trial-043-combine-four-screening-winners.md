# trial-043 — Комбинация четырёх screening-победителей

- Статус: **accepted for screening; better than control and best single feature**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Гипотеза

Четыре победителя одиночной серии могут дополнять друг друга: сильнейший share-
сигнал преимущественно улучшал exact-модель, а frequency уведомлений — base-
модель. Риск состоял в конфликте между компонентами, потому что два других
кандидата в одиночных trial ухудшали один из них.

## Единственное изменение

В exact- и base-компоненты одновременно добавлен фиксированный пакет:

```text
work_study_screen_share
entertainment_screen_share
notifications_per_day_value_frequency
daily_screen_time_hours_value_frequency
```

Обе frequency-фичи вычисляются fold-safe: mapping строится только по train-
части fold, unseen validation-значения получают `0`. Веса blend, folds, seed,
число деревьев и learning rate не менялись. Подбор состава или веса по OOF не
проводился.

## Результаты против контроля

| Вариант | Control OOF | Candidate OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.962825701 | 0.963720876 | **+0.000895176** |
| Base LightGBM | 0.959447155 | 0.960808070 | **+0.001360915** |
| 60/40 blend | 0.964538668 | **0.965239802** | **+0.000701134** |

Blend выиграл оба folds: `+0.000742039` и `+0.000659863`. Paired bootstrap на
100 повторах размера test полностью положителен:
`[+0.000592333; +0.000825531]`, положительных повторов `100/100`.

Сумма четырёх одиночных point-delta равнялась `+0.000707344`; фактическая
дельта пакета отличается всего на `-0.000006210`. Это диагностическое сравнение,
а не предположение об обязательной аддитивности AUC.

## Сравнение с лучшей одиночной фичей

Лучший одиночный `trial-029` имел OOF `0.964990697`. Пакет выше него на
`+0.000249104` и выиграл на обоих folds относительно одиночного лидера:
`+0.000233491` и `+0.000264738`.

Отдельный paired bootstrap `trial-043 - trial-029` на 300 повторах дал interval
`[+0.000167350; +0.000324207]`; положительны `300/300` повторов. Exact вырос
относительно лидера ещё на `+0.000089286`, base — на `+0.001367429`.

## Вывод

Комбинация **принята для screening**. Остальные три победителя добавили
устойчивый сигнал поверх `work_study_screen_share`, а base-компонент больше не
мешает общему результату. Перед переносом в финальный эксперимент пакет нужно
подтвердить на 5 folds. Target encoding и submission в этом trial не запускались.

## Команды

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-043 --candidate screening_winners_bundle
.\.venv\Scripts\python.exe -u src\compare_oof.py --control artifacts\oof_predictions_trial-029.csv --candidate artifacts\oof_predictions_trial-043.csv --prediction-column candidate_blend_fold_rank --repeats 300 --sample-size 296302 --seed 2026 --output artifacts\comparison_trial-043_vs_trial-029.json
```

Артефакты: `artifacts/metrics_trial-043_feature.json`, OOF, importance,
`artifacts/comparison_trial-043_vs_trial-029.json` и лог
`20260818T131331029531+0300_cv-feature-trial_trial-043.log`.
