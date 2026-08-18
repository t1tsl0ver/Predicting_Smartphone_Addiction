# trial-025 — Notifications per screen hour

- Статус: **rejected; mixed folds and paired interval crosses zero**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`
- Режим: быстрый 2-fold screening

## Вопрос

Добавляет ли интенсивность уведомлений на час экранного времени отдельный
поведенческий сигнал поверх абсолютных `notifications_per_day` и
`daily_screen_time_hours`?

## Гипотеза

Одинаковое количество уведомлений может означать разную нагрузку для человека с
двумя и десятью часами экрана. Отношение должно представить интенсивность, а не
общий объём использования. Предварительная OOF-диагностика показала, что эта
величина заметно отличается от ранга `exp-005`, но это было только основанием для
trial, а не доказательством прироста.

## Единственное основное изменение

В exact- и base-компоненты LightGBM добавлена одна числовая фича:

```text
notifications_per_screen_hour =
    notifications_per_day / daily_screen_time_hours
```

Нулевой или отрицательный знаменатель превращается в `NaN`; в фактических данных
минимальный ненулевой `daily_screen_time_hours` равен `0.5`. Exact-копия новой
фичи не создаётся.

## Что остаётся неизменным

```text
folds=2
shuffle=True
seed=2026
iterations=200
learning_rate=0.10
early_stopping=False
exact/base rank blend=60/40
control=cached trial-023 OOF on identical rows and folds
submission=None
```

Кандидат считается принятым только при положительной общей дельте, победе на
обоих folds и положительной нижней границе paired row-bootstrap интервала.

## Результаты

| Вариант | Контроль OOF | Кандидат OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.9628257006 | 0.9628386569 | +0.0000129564 |
| Base LightGBM | **0.9594471550** | 0.9590814346 | **-0.0003657204** |
| 60/40 blend | 0.9645386677 | **0.9645502646** | +0.0000115968 |

### Fold-wise blend

| Fold | Контроль | Кандидат | Delta |
|---:|---:|---:|---:|
| 1 | **0.9646319888** | 0.9646032162 | -0.0000287726 |
| 2 | 0.9644457185 | **0.9644982761** | +0.0000525577 |

### Paired bootstrap

100 stratified повторов размера test (`296302`):

```text
mean delta:          +0.0000152
95% interval:        [-0.0000235, +0.0000606]
positive repeats:    76/100
```

Интервал описывает только row-sampling uncertainty этой пары OOF predictions.
Он не учитывает смену folds, повторное обучение или leaderboard split.

Корреляция контрольного и кандидатного blend очень высокая: Spearman
`0.998581`. Новая фича получила mean gain `80.79` в exact-компоненте и `1755.57`
в base-компоненте, но использование split-ами не превратилось в устойчивое
улучшение AUC.

## Вывод

`notifications_per_screen_hour` **отклонена как одиночный кандидат**. Точечный
прирост blend слишком мал, folds имеют разные знаки, а paired interval пересекает
ноль. Особенно важно, что base-компонент заметно ухудшился: явное отношение не
добавило устойчивого ранжирования поверх исходных колонок.

Результат не запрещает будущую проверку взаимодействия в заранее согласованной
комбинации, но фича не входит в список одиночных победителей.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-025 --candidate notifications_per_screen_hour
```

Артефакты:

- `artifacts/metrics_trial-025_feature.json`;
- `artifacts/oof_predictions_trial-025.csv`;
- `artifacts/feature_importance_trial-025.csv`;
- `artifacts/logs/20260818T121928287150+0300_cv-feature-trial_trial-025.log`.
