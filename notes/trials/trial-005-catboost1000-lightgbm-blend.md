# trial-005 — CatBoost d7/1000 + LightGBM rank blend

- Статус: **blend rejected; effectively tied with LightGBM**
- Дата: 2026-08-13
- Контроль: LightGBM baseline из `trial-003`

## Вопрос

Может ли более сильный CatBoost d7/1000 дополнить LightGBM, хотя CatBoost d7/400
в `trial-003` только ухудшал его?

## Компоненты

| Модель | Local ROC AUC |
|---|---:|
| CatBoost d7/1000 | 0.9607046666 |
| LightGBM | **0.9628707172** |

CatBoost был повторно обучен только для сохранения построчных validation-
предсказаний и точно воспроизвёл результат `trial-001`. LightGBM повторно не
обучался: использованы сохранённые predictions на том же holdout.

## Разнообразие прогнозов

```text
Pearson correlation:  0.991840
Spearman correlation: 0.989218
```

По сравнению с CatBoost d7/400 Spearman-корреляция с LightGBM выросла с `0.982526`
до `0.989218`. Более длинный CatBoost стал сильнее, но одновременно похожее на
LightGBM по порядку объектов.

## Rank blends

| CatBoost | LightGBM | Validation AUC | Δ к LightGBM |
|---:|---:|---:|---:|
| 15% | 85% | **0.9628571201** | **-0.0000135971** |
| 30% | 70% | 0.9627326163 | -0.0001381009 |
| 50% | 50% | 0.9623970476 | -0.0004736696 |
| 70% | 30% | 0.9618651312 | -0.0010055860 |
| 85% | 15% | 0.9613392198 | -0.0015314974 |

## Вывод

Лучший blend практически равен LightGBM, но формально не превзошёл его. На этом
holdout нет основания включать CatBoost d7/1000 в ансамбль. Малый отрицательный
разрыв может быть нестабилен между folds, поэтому окончательный подбор весов
следует делать через out-of-fold predictions, а не добавлением всё новых весов на
том же validation.

Проверка d8/1000 оправдана: он сильнее d7/1000, его predictions уже сохранены, а
вычислительная стоимость нового blend близка к нулю.

## Воспроизведение

```powershell
.\.venv\Scripts\python.exe src\train.py --model catboost --iterations 1000 --depth 7 --skip-final --run-id trial-005-cat1000 --save-validation-predictions
.\.venv\Scripts\python.exe src\blend.py --primary artifacts\validation_predictions_trial-005-cat1000_catboost.csv --secondary artifacts\validation_predictions_trial-003_lightgbm.csv --primary-name catboost-d7-1000 --secondary-name lightgbm --run-id trial-005
```
