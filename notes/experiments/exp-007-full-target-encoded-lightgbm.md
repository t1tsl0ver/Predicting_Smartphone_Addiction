# exp-007 — Full target-encoded LightGBM blend

- Статус: **completed locally; retained as an `exp-008` component**
- Дата: 2026-08-18
- Источник признаков: полный принятый пакет `trial-045`
- Модель: 60/40 rank blend exact/base LightGBM

## Цель

Перенести все четыре screening-фичи и 15 leak-safe target encodings из
`trial-045` в полный 5-fold режим. Групповая чистка `trial-046` была отклонена,
поэтому ни одна TE-копия не удалялась.

## Конфигурация

```text
outer folds=5, shuffle=True, seed=2026
iterations=3000 maximum per fold
learning_rate=0.01
num_leaves=31
min_child_samples=20
early_stopping_rounds=400
exact/base rank weights=0.60/0.40
target encoding alpha=50
target encoding inner folds=5
```

Для model-train строк TE вычисляется inner-OOF. Validation и test кодируются
только статистиками соответствующего outer-train. Frequency-признаки также
строятся только по outer-train; неизвестные значения получают ноль.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_target_encoded_blend.py --run-id exp-007 --folds 5 --iterations 3000 --learning-rate 0.01 --early-stopping-rounds 400 --exact-weight 0.60 --target-encoding-alpha 50 --target-encoding-inner-folds 5 --log-every 100 --submission submission.csv
```

## Результат

| Компонент | Fold-rank OOF ROC AUC |
|---|---:|
| Exact LightGBM | 0.9657441343 |
| Base LightGBM | 0.9665233627 |
| `exp-007` 60/40 blend | **0.9668305140** |
| Контроль `exp-005` | **0.9668743680** |

Полный TE-кандидат практически повторил `exp-005`, но оказался ниже на
`0.0000438541`. Base-компонент стал сильнее exact-компонента: массовый TE
действительно передал ему значительную часть информации, которую раньше давали
exact-value категории.

| Fold | `exp-007` blend AUC |
|---:|---:|
| 1 | 0.9660790587 |
| 2 | 0.9671927750 |
| 3 | 0.9677635411 |
| 4 | 0.9671024750 |
| 5 | 0.9660126413 |

Полный запуск занял `2095` секунд, около 35 минут. Exact folds выбрали примерно
2415–2520 итераций; base почти везде дошёл до потолка 3000.

## Вывод

Как самостоятельная замена `exp-005` модель не принята. Однако её OOF/test
predictions сохранены: отличия в ошибках оказались достаточными, чтобы она стала
полезным компонентом финального post-hoc blend `exp-008`. Отдельный Kaggle
submission `exp-007` не загружался.
