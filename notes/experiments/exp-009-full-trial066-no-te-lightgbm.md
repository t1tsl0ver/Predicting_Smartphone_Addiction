# exp-009 — Full trial-066 no-TE LightGBM blend

- Статус: **completed locally; best standalone no-TE model**
- Дата: 2026-08-29
- Источник признаков: принятый семифичевый пакет `trial-066`
- Модель: 60/40 rank blend exact/base LightGBM

## Цель

Проверить, переносится ли сильный сигнал семи обычных engineered features из
лёгкого `trial-066` в полный бюджет. Это прямой no-TE эксперимент: никаких
target-dependent статистик, target encoding или decimal-digit фич не добавлено.

## Конфигурация

```text
outer folds=5, shuffle=True, seed=2026
iterations=3000 maximum per fold
learning_rate=0.01
num_leaves=31
min_child_samples=20
early_stopping_rounds=400
exact/base rank weights=0.60/0.40
target encoding=disabled
exact features=29
base features=20
```

Frequency-признаки строились только по outer-train каждого fold; неизвестные
validation/test значения получали ноль. Exact-вариант дополнительно использовал
categorical-копии девяти исходных числовых колонок.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_target_encoded_blend.py --run-id exp-009 --feature-bundle trial-066-no-te --folds 5 --iterations 3000 --learning-rate 0.01 --early-stopping-rounds 400 --exact-weight 0.60 --log-every 100
```

Submission намеренно не передавался: эксперимент сохранял только metrics,
OOF/test predictions и importance. Канонический `submission.csv` остался от
`exp-008`, SHA256 `a8f10cf749244e4423c6399039485ab37d3689698b9f2512c245a9d00ac8af40`.

## Результат

| Компонент | Fold-rank OOF ROC AUC |
|---|---:|
| Exact LightGBM | 0.9667012675 |
| Base LightGBM | 0.9637438759 |
| `exp-009` 60/40 blend | **0.9671371206** |

Тяжёлый бюджет улучшил тот же пакет относительно лёгкого `trial-066` на
**`+0.0004829658`**. Модель также превысила прежний no-TE ориентир `exp-005` на
**`+0.0002627533`**, но не обошла общий TE-ансамбль `exp-008`: разрыв
`-0.0002013897`.

| Fold | `exp-009` | Delta к `trial-066` | Delta к `exp-005` |
|---:|---:|---:|---:|
| 1 | 0.9663813912 | +0.0004514527 | +0.0003218633 |
| 2 | 0.9677213470 | +0.0004342606 | +0.0002970385 |
| 3 | 0.9679543582 | +0.0005959529 | +0.0002366693 |
| 4 | 0.9672593684 | +0.0004658525 | +0.0001428210 |
| 5 | 0.9663633981 | +0.0004649663 | +0.0003123970 |

Выигрыш получен на 5/5 folds против обоих no-TE контролей.

## Paired bootstrap

Во всех сравнениях использованы 300 стратифицированных парных выборок размера
Kaggle test (`296302`), seed `2026`.

| Контроль | OOF delta | Mean bootstrap delta | 95% interval | P(delta > 0) |
|---|---:|---:|---:|---:|
| `trial-066` | +0.0004829658 | +0.0004856878 | [+0.0004380893; +0.0005489756] | 1.00 |
| `exp-005` | +0.0002627533 | +0.0002614021 | [+0.0001594656; +0.0003794344] | 1.00 |
| `exp-008` | -0.0002013897 | -0.0002003078 | [-0.0003037236; -0.0000985008] | 0.00 |

Интервалы учитывают только неопределённость от выбора строк, но не обучение,
выбор feature bundle или leaderboard split.

## Технические наблюдения

- Exact folds выбирали `1786–1990` деревьев: early stopping нашёл стабильное
  плато до лимита.
- Base дошёл до `2999–3000` на всех folds и продолжал улучшаться у границы.
  Это сигнал для отдельной гипотезы о большем base-бюджете, но не доказательство
  будущего OOF-прироста.
- Среди engineered features наибольший gain получили `social_screen_share`,
  `notifications_per_day_value_frequency`, `work_study_screen_share` и
  `gaming_screen_share`. Importance не заменяет отдельные heavy ablation.
- Полный CV занял `1381` секунду, весь запуск с загрузкой и записью — около
  23 минут.
- Проверены `691369` уникальных OOF-строк и `296302` test-строки; row/fold/target
  alignment совпал с контролями, ID test совпали, все predictions конечны.

## Артефакты

```text
artifacts/metrics_exp-009_full_feature_blend.json
artifacts/oof_predictions_exp-009.csv
artifacts/test_predictions_exp-009.csv
artifacts/feature_importance_exp-009.csv
artifacts/comparison_exp-009_vs_trial-066.json
artifacts/comparison_exp-009_vs_exp-005.json
artifacts/comparison_exp-009_vs_exp-008.json
```

## Вывод

Гипотеза подтверждена: принятые обычные features сохраняют и усиливают сигнал в
полном режиме. `exp-009` — новый лучший автономный no-TE кандидат, устойчиво
сильнее `trial-066` и `exp-005`. Общим лидером остаётся `exp-008`.

Сохранённые OOF/test predictions делают возможной дешёвую post-hoc проверку:
заменить старую no-TE часть ансамбля новой `exp-009` или проверить её добавочный
residual-сигнал. Это новый эксперимент и запускается только после отдельного
согласования; веса по Public LB не подбираются.
