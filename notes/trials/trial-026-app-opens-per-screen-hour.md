# trial-026 — App opens per screen hour

- Статус: **rejected; negative point estimate and mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`
- Режим: быстрый 2-fold screening

## Вопрос

Добавляет ли число открытий приложений на час экранного времени отдельный сигнал
интенсивности взаимодействия поверх абсолютных `app_opens_per_day` и
`daily_screen_time_hours`?

## Гипотеза

При одинаковом числе открытий более короткое экранное время может означать более
частое переключение между приложениями. Явное отношение может быть удобнее для
дерева, чем самостоятельное восстановление взаимодействия двух колонок.

## Единственное основное изменение

В exact- и base-компоненты LightGBM добавлена одна числовая фича:

```text
app_opens_per_screen_hour =
    app_opens_per_day / daily_screen_time_hours
```

Нулевой или отрицательный знаменатель превращается в `NaN`. Exact-копия новой
фичи не создаётся.

## Неизменная проверка

```text
folds=2, shuffle=True, seed=2026
iterations=200, learning_rate=0.10, early_stopping=False
exact/base fold-rank blend=60/40
control=cached trial-023 OOF on identical rows and folds
submission=None
```

## Результаты

| Вариант | Контроль OOF | Кандидат OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | **0.9628257006** | 0.9628124316 | -0.0000132689 |
| Base LightGBM | **0.9594471550** | 0.9591278219 | -0.0003193331 |
| 60/40 blend | **0.9645386677** | 0.9645153237 | -0.0000233440 |

### Fold-wise blend

| Fold | Контроль | Кандидат | Delta |
|---:|---:|---:|---:|
| 1 | 0.9646319888 | **0.9646354540** | +0.0000034651 |
| 2 | **0.9644457185** | 0.9643957468 | -0.0000499717 |

### Paired bootstrap

```text
repeats:             100
sample size:         296302
mean delta:          -0.0000235
95% interval:        [-0.0000752, +0.0000230]
positive repeats:    18/100
```

Spearman-корреляция контрольного и кандидатного blend равна `0.998380`. Новая
фича получила mean gain `44.53` в exact-компоненте и `1496.20` в base, но оба
компонента ухудшили OOF.

## Вывод

`app_opens_per_screen_hour` **отклонена как одиночный кандидат**. Общая дельта
отрицательна, folds имеют разные знаки, bootstrap пересекает ноль. Как и в
`trial-025`, явное отношение особенно навредило более слабому base-компоненту.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-026 --candidate app_opens_per_screen_hour
```

Артефакты:

- `artifacts/metrics_trial-026_feature.json`;
- `artifacts/oof_predictions_trial-026.csv`;
- `artifacts/feature_importance_trial-026.csv`;
- `artifacts/logs/20260818T122428565728+0300_cv-feature-trial_trial-026.log`.
