# trial-028 — Entertainment screen share

- Статус: **accepted for screening; positive paired OOF and won 2/2 folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`
- Режим: быстрый 2-fold screening

## Вопрос и гипотеза

Добавляет ли доля развлекательного экранного времени новый композиционный
сигнал поверх абсолютных daily, social-media и gaming часов? При одинаковом
общем времени различный состав использования может соответствовать разному
риску, а явная доля упрощает это взаимодействие для дерева.

## Единственное основное изменение

В оба LightGBM-компонента добавлена одна числовая фича без exact-копии:

```text
entertainment_screen_share =
    (social_media_hours + gaming_hours) / daily_screen_time_hours
```

Неположительный знаменатель превращается в `NaN`. Folds, seed, 200 деревьев,
learning rate `0.10`, exact/base rank blend `60/40` и cached контроль совпадают
с `trial-023`. Submission не создаётся.

## Результаты

| Вариант | Контроль OOF | Кандидат OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.9628257006 | **0.9629947416** | **+0.0001690410** |
| Base LightGBM | **0.9594471550** | 0.9592964506 | -0.0001507044 |
| 60/40 blend | 0.9645386677 | **0.9646571249** | **+0.0001184572** |

### Fold-wise blend

| Fold | Контроль | Кандидат | Delta |
|---:|---:|---:|---:|
| 1 | 0.9646319888 | **0.9647896062** | **+0.0001576174** |
| 2 | 0.9644457185 | **0.9645251030** | **+0.0000793845** |

### Paired bootstrap

```text
repeats:             100
sample size:         296302
95% interval:        [+0.0000402, +0.0002175]
positive repeats:    100/100
```

Spearman control/candidate blend равен `0.996979`. Mean gain новой фичи:
`10989.59` в exact и `12960.96` в base-компоненте. Base активно использовал
признак, но его собственный OOF снизился; общий прирост обеспечили exact и
полезное изменение итогового blend-ранжирования.

## Вывод

`entertainment_screen_share` **принята как одиночный screening-кандидат**:
положительная общая дельта, победа 2/2 folds и paired interval полностью выше
нуля. Это пока не 5-fold подтверждение и не разрешение автоматически включать
фичу в комбинации. До завершения одиночной серии она сохраняется в рейтинге
отдельным кандидатом.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-028 --candidate entertainment_screen_share
```

Артефакты: `artifacts/metrics_trial-028_feature.json`,
`artifacts/oof_predictions_trial-028.csv`,
`artifacts/feature_importance_trial-028.csv` и лог
`20260818T122926241195+0300_cv-feature-trial_trial-028.log`.
