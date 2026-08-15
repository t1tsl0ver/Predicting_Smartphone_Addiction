# trial-020 — Stronger categorical smoothing

- Статус: **rejected; negative point estimate and inconclusive paired interval**
- Дата: 2026-08-15
- Контроль: `trial-019`

## Вопрос

Сможет ли более сильное сглаживание categorical splits уменьшить заметное
переобучение exact-value categories, сохранив их полезный сигнал?

## Гипотеза

После добавления exact-value categories mean train AUC быстрого контроля вырос
до `0.99459`, тогда как validation остался около `0.96403`. LightGBM
`cat_smooth` регулирует доверие к статистике категориальных значений. Умеренное
увеличение параметра может снизить подгонку под специфические exact-значения.

## Единственное основное изменение

```text
cat_smooth: 10.0 (LightGBM default) -> 20.0
```

Всё остальное совпадает с `trial-019`:

```text
model=LGBMClassifier
n_estimators=500
learning_rate=0.1
num_leaves=31
max_depth=-1
min_child_samples=20
reg_alpha=0.0
reg_lambda=0.0
early_stopping_rounds=0
folds=5
shuffle=True
random_state=2026
numeric_exact_categories=True
```

В `src/cv_lightgbm.py` добавлен явный CLI-параметр `--cat-smooth`; он
передаётся в модель, выводится в лог и сохраняется в JSON-метриках.

## Команды

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id trial-020 --cat-smooth 20 --drop-features gender stress_level --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep --add-numeric-exact-categories

.\.venv\Scripts\python.exe -u src\compare_oof.py --control artifacts\oof_predictions_trial-019.csv --candidate artifacts\oof_predictions_trial-020.csv --prediction-column lightgbm_probability --repeats 300 --sample-size 296302 --seed 2026 --output artifacts\comparison_trial-019_vs_trial-020.json
```

## Результат

| Метрика | `trial-019` | `trial-020` | Delta |
|---|---:|---:|---:|
| Raw OOF ROC AUC | **0.9640249964** | 0.9639773146 | **-0.0000476818** |
| Fold-rank OOF ROC AUC | **0.9640257010** | 0.9639788420 | **-0.0000468590** |

Для внутренних test predictions выбран fold-rank вариант. Submission не
создавался.

### Сравнение по folds

| Fold | `trial-019` | `trial-020` | Delta |
|---:|---:|---:|---:|
| 1 | 0.9631629673 | **0.9632147095** | +0.0000517422 |
| 2 | **0.9647113599** | 0.9645324607 | -0.0001788992 |
| 3 | 0.9647552505 | **0.9647720064** | +0.0000167559 |
| 4 | **0.9643327627** | 0.9641161597 | -0.0002166030 |
| 5 | 0.9631661562 | **0.9632588665** | +0.0000927103 |

Изменение выиграло 3/5 folds, но два проигрыша оказались больше трёх выигрышей.
Mean train AUC снизился с `0.99459` до `0.99335`, то есть сглаживание реально
уменьшило подгонку, однако validation от этого не улучшился.

### Paired bootstrap raw OOF delta

```text
prediction Pearson correlation:  0.990437
prediction Spearman correlation: 0.992787
point delta:                      -0.0000477
mean bootstrap delta:             -0.0000462
standard deviation:               0.0000716
paired 95% interval:              [-0.0001814, +0.0000848]
positive repeats:                 27%
```

Интервал пересекает ноль: отличие мало относительно row-sampling uncertainty.
Тем не менее точечный общий OOF отрицателен, поэтому оснований принимать
изменение нет.

## Время

- CV: `2:16`;
- полный запуск: `2:26`.

## Решение

`cat_smooth=20` **отклонён**. Активным контролем остаётся `trial-019` с
дефолтным `cat_smooth=10`. Результат не доказывает, что exact categories не
переобучаются; он показывает только, что дополнительное categorical smoothing
не конвертировало уменьшение train AUC в прирост validation AUC.

Артефакты запуска:

- `artifacts/metrics_trial-020_cv.json`;
- `artifacts/oof_predictions_trial-020.csv`;
- `artifacts/test_predictions_trial-020.csv`;
- `artifacts/feature_importance_trial-020.csv`;
- `artifacts/comparison_trial-019_vs_trial-020.json`;
- `artifacts/logs/20260815T135659198268+0300_cv-lightgbm_trial-020.log`.
