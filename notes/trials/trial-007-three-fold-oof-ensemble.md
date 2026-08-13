# trial-007 — Three-fold OOF ensemble confirmation

- Статус: **accepted; submission created**
- Дата: 2026-08-13
- Контроль: LightGBM baseline внутри тех же CV folds

## Вопрос

Является ли маленький holdout-прирост смеси 15% CatBoost d8/1000 и 85% LightGBM
устойчивым на нескольких независимых validation folds?

## Заранее зафиксированный дизайн

- `StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)`;
- CatBoost: depth 8, максимум 1000 деревьев;
- LightGBM: та же конфигурация, что в `trial-003`, максимум 3000 деревьев;
- единственный проверяемый вес: 15% CatBoost + 85% LightGBM;
- внутри каждого fold смешиваются процентные ранги predictions;
- blend принимается, если средний fold delta положительный и он выигрывает не
  менее двух folds из трёх;
- веса после просмотра результатов не меняются.

Каждая строка train получила prediction модели, которая не использовала её при
обучении. Test predictions каждой модели усреднены по трём folds.

## Результаты folds

| Fold | CatBoost | LightGBM | Blend | Δ blend к LGBM |
|---:|---:|---:|---:|---:|
| 1 | 0.9617424 | 0.9632249 | **0.9632757** | +0.0000508 |
| 2 | 0.9620452 | 0.9633594 | **0.9634458** | +0.0000864 |
| 3 | 0.9614216 | 0.9626541 | **0.9627412** | +0.0000871 |

Blend выиграл `3/3` folds. Средний fold delta положителен, поэтому заранее
заданное правило выполнено.

## Общие OOF-метрики

| Предсказание | OOF ROC AUC |
|---|---:|
| CatBoost raw probabilities | 0.9617348260 |
| LightGBM raw probabilities | 0.9630781377 |
| LightGBM fold ranks | 0.9630794609 |
| 15/85 blend fold ranks | **0.9631544070** |

OOF-прирост blend относительно LightGBM fold ranks: `+0.0000749461`.

Абсолютный прирост мал, но направление стабильно на всех folds. Это значительно
сильнее предыдущего подтверждения на одном многократно использованном holdout.

## Train и validation

| Fold | Cat train AUC | Cat valid AUC | LGBM train AUC | LGBM valid AUC |
|---:|---:|---:|---:|---:|
| 1 | 0.9654385 | 0.9617424 | 0.9716236 | 0.9632249 |
| 2 | 0.9650783 | 0.9620452 | 0.9712088 | 0.9633594 |
| 3 | 0.9654717 | 0.9614216 | 0.9718398 | 0.9626541 |

Training AUC выше validation, особенно у LightGBM, но validation AUC продолжал
улучшаться почти до верхней границы деревьев. CatBoost использовал 1000 деревьев
на каждом fold; лучшие LightGBM iterations: 2998, 2934 и 2999.

## Выбранный submission

```text
15% среднего CatBoost test fold-rank
+
85% среднего LightGBM test fold-rank
```

Submission:

```text
rows=296302
minimum=0.0000177184
maximum=0.9999061768
sha256=6351318d403cdf541d655217e556792d12d63ac03e1d0baa9daad513bb13ef70
```

Kaggle score будет добавлен после загрузки участником. До этого `exp-001` не
фиксируется.

## Время

Полный 3-fold запуск занял около 30 минут 10 секунд. Он включал шесть обучений,
OOF/test inference и сохранение predictions.

## Воспроизведение

```powershell
.\.venv\Scripts\python.exe src\cv_ensemble.py --run-id trial-007
.\.venv\Scripts\python.exe src\create_cv_submission.py
```
