# trial-045 — Target encoding всех текущих смысловых фич

- Статус: **accepted for screening; won 2/2 folds against trial-043**
- Дата: 2026-08-18
- Контроль: пакет четырёх screening-победителей `trial-043`

## Гипотеза

Одиночный target encoding notifications/day не улучшил blend, но заметно поднял
base-компонент. Одновременное supervised-представление всего профиля может дать
base-модели целостный набор совместимых сигналов, а не одну фичу, конфликтующую
с остальным feature space.

## Единственное изменение

Поверх неизменного пакета `trial-043` добавлены target-encoded копии всех 15
уникальных смысловых фич текущего base-набора:

- 9 исходных числовых признаков;
- `academic_work_impact`;
- `work_study_screen_share` и `entertainment_screen_share`;
- `other_screen_time_hours`, `screen_social_load` и
  `high_stress_other_screen_per_sleep`.

Категориальные `*_exact` не кодировались повторно, потому что дублируют исходные
значения. Две `*_value_frequency` также исключены: это уже encoding исходных
столбцов, а не отдельные смысловые источники.

Для всех 15 копий использован один заранее фиксированный leak-safe протокол:

- outer validation mapping только по outer-train;
- 5-fold inner-OOF encoding для строк модельного train;
- smoothing `alpha=50`;
- missing как отдельное значение;
- unseen получает global target mean mapping-train.

Собственный target строки не участвует в её encoding. Модель, веса 60/40,
outer folds, seed и остальные фичи не менялись.

## Результаты против trial-043

| Вариант | Control OOF | Candidate OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.963720876 | 0.963739498 | +0.000018622 |
| Base LightGBM | 0.960808070 | **0.965155402** | **+0.004347331** |
| 60/40 blend | 0.965239802 | **0.965571690** | **+0.000331888** |

Blend выиграл оба folds: `+0.000289370` и `+0.000374928`. Base тоже выиграл
оба folds на `+0.003973627` и `+0.004721039`; exact был смешан
(`-0.000026019`, `+0.000063263`).

Основной 100-repeat paired bootstrap полностью положителен:
`[+0.000214920; +0.000466602]`. Дополнительный 300-repeat bootstrap подтвердил
результат: `[+0.000189744; +0.000469104]`, положительны `300/300` повторов.

## Какие encoding использовала модель

Наибольший mean gain среди TE-копий:

| Target encoding | Base gain | Exact gain |
|---|---:|---:|
| `daily_screen_time_hours` | 621318 | 590872 |
| `weekend_screen_time` | 182081 | 176679 |
| `notifications_per_day` | 111726 | 76981 |
| `app_opens_per_day` | 94524 | 54573 |
| `social_media_hours` | 42307 | 42902 |

Большинство высококардинальных engineered TE-копий получили очень малый gain;
`entertainment_screen_share_target_encoding` в exact не использовался вообще.
Поэтому успешный пакет не доказывает полезность каждой из 15 фич по отдельности.

## Вывод

Массовый target encoding **принят для screening**. В отличие от одиночного
`trial-044`, согласованный набор supervised-признаков радикально усилил base-
модель и дал устойчивый прирост итоговому blend. Это практическая синергия
набора, но состав ещё не оптимизирован: часть TE-копий, вероятно, лишняя.

Перед финальным exp нужны отдельное решение о возможных ablation и затем 5-fold
подтверждение. Текущий 2-fold OOF нельзя напрямую сравнивать с high-budget
`exp-005`. Submission не создавался.

## Команды

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-045 --candidate screening_winners_plus_all_target_encodings --control-oof artifacts\oof_predictions_trial-043.csv --control-metrics artifacts\metrics_trial-043_feature.json --target-encoding-alpha 50 --target-encoding-inner-folds 5
.\.venv\Scripts\python.exe -u src\compare_oof.py --control artifacts\oof_predictions_trial-043.csv --candidate artifacts\oof_predictions_trial-045.csv --prediction-column candidate_blend_fold_rank --repeats 300 --sample-size 296302 --seed 2026 --output artifacts\comparison_trial-045_vs_trial-043.json
```

Артефакты: `artifacts/metrics_trial-045_feature.json`, OOF, importance,
`artifacts/comparison_trial-045_vs_trial-043.json` и лог
`20260818T135459332270+0300_cv-feature-trial_trial-045.log`.
