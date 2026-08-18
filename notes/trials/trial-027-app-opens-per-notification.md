# trial-027 — App opens per notification

- Статус: **rejected; effectively zero delta and mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`
- Режим: быстрый 2-fold screening

## Вопрос и гипотеза

Добавляет ли условный отклик на уведомления новый поведенческий сигнал поверх
абсолютных `app_opens_per_day` и `notifications_per_day`? Число открытий на одно
уведомление может описывать реакцию пользователя, а не общий объём активности.

## Единственное основное изменение

В оба LightGBM-компонента добавлена одна числовая фича без exact-копии:

```text
app_opens_per_notification =
    app_opens_per_day / notifications_per_day
```

Неположительный знаменатель превращается в `NaN`. Folds, seed, 200 деревьев,
learning rate `0.10`, exact/base rank blend `60/40` и cached контроль полностью
совпадают с предыдущими одиночными trial. Submission не создаётся.

## Результаты

| Вариант | Контроль OOF | Кандидат OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.9628257006 | **0.9628621080** | +0.0000364074 |
| Base LightGBM | **0.9594471550** | 0.9591107407 | -0.0003364143 |
| 60/40 blend | 0.9645386677 | 0.9645392902 | +0.0000006224 |

Fold-deltas blend: `-0.0000267067` и `+0.0000282375`.

Paired bootstrap на 100 повторах размера test:

```text
95% interval:      [-0.0000365, +0.0000367]
positive repeats:  48/100
```

Spearman между control и candidate blend равен `0.999005`. Mean gain новой фичи
равен `70.70` в exact и `6980.90` в base-компоненте. Высокий base gain не дал
улучшения: base OOF заметно снизился.

## Вывод

`app_opens_per_notification` **отклонена как одиночный кандидат**. Общая дельта
практически равна нулю, folds расходятся по знаку, paired interval симметрично
пересекает ноль. Маленький exact-прирост полностью съеден ухудшением base.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-027 --candidate app_opens_per_notification
```

Артефакты: `artifacts/metrics_trial-027_feature.json`,
`artifacts/oof_predictions_trial-027.csv`,
`artifacts/feature_importance_trial-027.csv` и лог
`20260818T122708566015+0300_cv-feature-trial_trial-027.log`.
