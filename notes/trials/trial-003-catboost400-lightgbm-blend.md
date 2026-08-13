# trial-003 — CatBoost 400 + LightGBM rank blend

- Статус: **LightGBM accepted; tested blend rejected**
- Дата: 2026-08-13
- Контроль: CatBoost depth 7, 400 деревьев из `exp-000`

## Вопрос

Может ли независимый LightGBM дополнить дешёвый CatBoost d7/400 и дать ансамбль
с AUC выше одиночного CatBoost?

## Гипотеза

LightGBM строит деревья иначе, чем CatBoost, и может ранжировать часть объектов
по-другому. Смешивание рангов должно компенсировать часть ошибок CatBoost.

## Компоненты

### CatBoost

Контрольная конфигурация `exp-000`:

```text
iterations=400
depth=7
learning_rate=0.08
l2_leaf_reg=5.0
```

Повторное обучение с сохранением validation-прогнозов точно воспроизвело AUC
`0.9550075136`.

### LightGBM baseline

```text
n_estimators=3000
learning_rate=0.03
num_leaves=31
min_child_samples=100
reg_lambda=5.0
early_stopping_rounds=100
random_seed=42
```

LightGBM использовал те же 12 признаков и ровно тот же stratified holdout. Числовые
пропуски обрабатывались нативно, строковые колонки передавались как категории.
Систематический подбор параметров не проводился.

## Смешивание

Вероятности каждой модели заменялись процентными рангами, после чего ранги
усреднялись. Изначально были запланированы веса CatBoost/LightGBM `50/50`, `70/30`
и `85/15`. После неожиданного превосходства LightGBM симметрично добавлены
диагностические веса `30/70` и `15/85`; это адаптивный анализ после просмотра
результата, а не часть первоначального плана.

## Результаты отдельных моделей

| Модель | Local ROC AUC |
|---|---:|
| CatBoost d7/400 | 0.9550075 |
| LightGBM | **0.9628707** |

LightGBM превзошёл CatBoost d7/400 на `+0.0078632` и CatBoost d7/1000 из
`trial-001` на `+0.0021661`.

LightGBM дошёл до верхней границы 3000 деревьев; лучшая точка — 2999. Явного
плато по AUC не найдено, хотя прирост к концу стал мал.

## Разнообразие прогнозов

```text
Pearson correlation:  0.979923
Spearman correlation: 0.982526
```

Модели не идентичны, однако их ранжирование сильно коррелирует.

## Rank blends

| CatBoost | LightGBM | Validation AUC | Δ к CatBoost | Δ к LightGBM |
|---:|---:|---:|---:|---:|
| 15% | 85% | **0.9624236** | +0.0074161 | -0.0004471 |
| 30% | 70% | 0.9617152 | +0.0067077 | -0.0011555 |
| 50% | 50% | 0.9603632 | +0.0053556 | -0.0025076 |
| 70% | 30% | 0.9585538 | +0.0035463 | -0.0043169 |
| 85% | 15% | 0.9569012 | +0.0018937 | -0.0059695 |

## Вывод

Ансамбль действительно лучше контрольного CatBoost 400, но ни один проверенный
вес не лучше одного LightGBM. Следовательно, улучшение обеспечено LightGBM, а
добавление слабого CatBoost 400 ухудшает нового лидера.

- LightGBM принимается как сильный самостоятельный кандидат.
- Ансамбль CatBoost d7/400 + LightGBM отклоняется.
- Гипотеза о пользе ансамбля с более сильным CatBoost d7/1000 остаётся открытой
  и проверяется отдельно, как было согласовано заранее.

## Воспроизведение

```powershell
.\.venv\Scripts\python.exe src\train.py --model catboost --iterations 400 --depth 7 --skip-final --run-id trial-003-cat400 --save-validation-predictions
.\.venv\Scripts\python.exe src\train_lightgbm.py --run-id trial-003
.\.venv\Scripts\python.exe src\blend.py --primary artifacts\validation_predictions_trial-003-cat400_catboost.csv --secondary artifacts\validation_predictions_trial-003_lightgbm.csv --primary-name catboost-d7-400 --secondary-name lightgbm --run-id trial-003
```
