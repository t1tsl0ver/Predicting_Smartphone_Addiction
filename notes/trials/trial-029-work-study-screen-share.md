# trial-029 — Work/study screen share

- Статус: **accepted for screening; strong paired OOF and won 2/2 folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`
- Режим: быстрый 2-fold screening

## Вопрос и гипотеза

Добавляет ли доля рабочего/учебного экранного времени новый композиционный
сигнал поверх абсолютных `work_study_hours` и `daily_screen_time_hours`? При
одинаковом общем времени высокий объём work/study и высокая его доля описывают
разные профили поведения.

## Единственное основное изменение

В оба LightGBM-компонента добавлена одна числовая фича без exact-копии:

```text
work_study_screen_share =
    work_study_hours / daily_screen_time_hours
```

Неположительный знаменатель превращается в `NaN`. Folds, seed, 200 деревьев,
learning rate `0.10`, exact/base rank blend `60/40` и cached контроль совпадают
с `trial-023`. Submission не создаётся.

## Результаты

| Вариант | Контроль OOF | Кандидат OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.9628257006 | **0.9636315903** | **+0.0008058898** |
| Base LightGBM | **0.9594471550** | 0.9594406415 | -0.0000065135 |
| 60/40 blend | 0.9645386677 | **0.9649906975** | **+0.0004520298** |

### Fold-wise blend

| Fold | Контроль | Кандидат | Delta |
|---:|---:|---:|---:|
| 1 | 0.9646319888 | **0.9651405365** | **+0.0005085477** |
| 2 | 0.9644457185 | **0.9648408435** | **+0.0003951250** |

### Paired bootstrap

```text
repeats:             100
sample size:         296302
95% interval:        [+0.0003472, +0.0005536]
positive repeats:    100/100
```

Spearman control/candidate blend равен `0.995939`. Mean gain новой фичи:
`20610.35` в exact и `43608.64` в base-компоненте. В отличие от предыдущих
ratio-trial, base остался практически нейтральным, а exact получил крупный
устойчивый прирост.

## Вывод

`work_study_screen_share` **принята как сильный одиночный screening-кандидат**:
общая дельта существенно выше noise floor, оба folds положительны, paired
interval полностью выше нуля. На момент завершения это первое место одиночной
серии. Пятифолдового подтверждения и добавления в комбинации пока нет.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-029 --candidate work_study_screen_share
```

Артефакты: `artifacts/metrics_trial-029_feature.json`,
`artifacts/oof_predictions_trial-029.csv`,
`artifacts/feature_importance_trial-029.csv` и лог
`20260818T123228730599+0300_cv-feature-trial_trial-029.log`.
