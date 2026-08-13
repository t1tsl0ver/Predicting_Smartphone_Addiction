# exp-001 — Three-fold gradient boosting ensemble

- Статус: **completed**
- Дата: 2026-08-13
- Kaggle score: **0.96466**
- Предыдущий Kaggle score: `0.95663`
- Leaderboard delta: **+0.00803**

## Итоговая конфигурация

Три стратифицированных folds с `shuffle=True`, `random_state=2026`. На каждом fold:

- CatBoost: depth 8, 1000 деревьев, learning rate 0.08;
- LightGBM: до 3000 деревьев, learning rate 0.03, 31 лист;
- test predictions каждой модели переводятся в процентные ранги;
- итог fold-моделей усредняется;
- финальный прогноз: 15% CatBoost ranks + 85% LightGBM ranks.

## Какие trial привели к версии

- `trial-001`: 1000 деревьев существенно улучшили CatBoost;
- `trial-002` и `trial-004`: depth 8 подтвердил преимущество над depth 7;
- `trial-003`: LightGBM стал лучшей одиночной моделью;
- `trial-005`: d7/1000 не улучшил LightGBM в смеси;
- `trial-006`: d8/1000 дал слабый положительный holdout-сигнал;
- `trial-007`: фиксированный вес 15/85 выиграл у LightGBM на всех 3 folds.

## OOF-подтверждение

| Fold | LightGBM | Blend | Δ blend |
|---:|---:|---:|---:|
| 1 | 0.9632249 | 0.9632757 | +0.0000508 |
| 2 | 0.9633594 | 0.9634458 | +0.0000864 |
| 3 | 0.9626541 | 0.9627412 | +0.0000871 |

```text
LightGBM fold-rank OOF AUC: 0.9630794609
Blend fold-rank OOF AUC:    0.9631544070
OOF delta:                 +0.0000749461
Fold wins:                  3/3
```

## Submission

```text
rows=296302
sha256=6351318d403cdf541d655217e556792d12d63ac03e1d0baa9daad513bb13ef70
Kaggle score=0.96466
```

Хеш связывает leaderboard score именно с submission, созданным в `trial-007`.

## Интерпретация

Основной leaderboard-прирост относительно `exp-000` пришёл не только от финальной
смеси: были увеличены бюджет и глубина CatBoost, добавлен более сильный LightGBM,
а test predictions усреднены по трём folds. Изолированный вклад CatBoost в
финальный blend мал — около `+0.000075` OOF AUC относительно LightGBM.

CatBoost и LightGBM являются градиентными бустингами, а Spearman-корреляция их
validation predictions около `0.99`. Поэтому значительный дополнительный прирост
от дальнейшего смешивания только этих двух семейств маловероятен.

## Следующие направления

- ablation исходных слабых признаков;
- осмысленный feature engineering;
- более ортогональная модель, которая ошибается иначе, но остаётся достаточно
  сильной самостоятельно;
- подбор параметров через CV после формирования разумного пространства поиска.

Подробный план находится в `notes/NEXT_DIRECTIONS.md`.

## Воспроизведение

```powershell
.\.venv\Scripts\python.exe src\cv_ensemble.py --run-id trial-007
.\.venv\Scripts\python.exe src\create_cv_submission.py
```
