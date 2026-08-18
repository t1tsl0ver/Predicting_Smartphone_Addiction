# exp-006 — Full exact-value CatBoost

- Статус: **completed; best standalone, rejected vs `exp-005`**
- Дата локального запуска: 2026-08-17
- Kaggle score получен: 2026-08-18
- Контроли: `trial-022`, `exp-003` и `exp-005`
- Модель: один CatBoost; LightGBM заново не обучался

## Цель

Перенести победивший в `trial-022` CatBoost в полноценный режим, аналогичный
лучшей low-learning-rate конфигурации `exp-003`, и проверить, сможет ли одна
CatBoost-модель обойти rank blend двух LightGBM из `exp-005`.

Это не механическая копия параметров LightGBM. Общий режим перенесён по смыслу:
5 folds, низкий learning rate, большой потолок деревьев, individual best
iteration и early stopping. Ёмкость и регуляризация оставлены CatBoost-native:
`depth=8` и `l2_leaf_reg=5`, подтверждённые предыдущими CatBoost trial.

## Конфигурация

```text
folds=5
shuffle=True
seed=2026
iterations=3000 maximum per fold
learning_rate=0.01
depth=8
l2_leaf_reg=5.0
early_stopping_rounds=400
use_best_model=True
bootstrap_type=MVS
subsample=0.8
random_strength=1.0
border_count=254
max_ctr_complexity=1
leaf_estimation_iterations=10
boosting_type=Plain
```

Использован принятый набор производных и exact-value признаков, но без
`gaming_hours_exact` по screening-сигналу `trial-024`: 21 признак, из них 9
категориальных. Сырая числовая колонка `gaming_hours` сохранена.

Важная оговорка: эксперимент проверяет весь пакет «полный режим CatBoost плюс
удалённая exact-копия». Он не изолирует 5-fold вклад удаления
`gaming_hours_exact` от изменения бюджета и learning rate.

## Реализация

Добавлен `src/cv_catboost.py`. Runner:

- проверяет train/test/sample submission и совпадение cached OOF controls;
- строит те же 5 folds с seed 2026;
- обучает только CatBoost;
- сохраняет prediction checkpoint после каждого fold;
- использует CatBoost snapshot внутри долгого fold;
- при повторном запуске проверяет fingerprint данных и конфигурации и пропускает
  только совместимые завершённые folds;
- сравнивает raw probabilities и fold ranks и выбирает вариант с большим OOF;
- выполняет paired stratified bootstrap против `exp-005` и `trial-022`;
- создаёт OOF/test artifacts, importance, канонический submission и metadata.

Команда:

```powershell
.\.venv\Scripts\python.exe -u src\cv_catboost.py
```

## Результат

| Модель | OOF ROC AUC | Delta к `exp-006` |
|---|---:|---:|
| `trial-022` CatBoost/200 | 0.9650517759 | **-0.0008439927** |
| `exp-003` standalone LightGBM | 0.9650442019 | **-0.0008515667** |
| `exp-006` raw CatBoost | 0.9658915738 | -0.0000041947 |
| `exp-006` fold-rank CatBoost | **0.9658957686** | — |
| `exp-005` LightGBM rank blend | **0.9668743680** | **+0.0009785994** |

Для submission выбраны усреднённые fold percentile ranks.

### Fold-wise сравнение

| Fold | `exp-003` | `exp-006` CatBoost | Delta к `exp-003` | `exp-005` | CatBoost − `exp-005` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9642666519 | **0.9651220870** | +0.0008554351 | **0.9660595286** | -0.0009374416 |
| 2 | 0.9655261857 | **0.9663541316** | +0.0008279459 | **0.9674243119** | -0.0010701804 |
| 3 | 0.9659118632 | **0.9667140182** | +0.0008021550 | **0.9677176869** | -0.0010036687 |
| 4 | 0.9653225469 | **0.9662295627** | +0.0009070158 | **0.9671165479** | -0.0008869852 |
| 5 | 0.9641937531 | **0.9650590350** | +0.0008652819 | **0.9660510005** | -0.0009919654 |

CatBoost выиграл у standalone `exp-003` на 5/5 folds, но проиграл бленду
`exp-005` на 5/5. Размер обеих дельт устойчив между folds.

## Paired bootstrap

300 stratified bootstrap-выборок размера test (`296302`), fold-rank predictions:

| Сравнение (`exp-006` minus control) | Point delta | 95% interval | Положительные повторы |
|---|---:|---:|---:|
| против `trial-022` CatBoost/200 | +0.0008440 | [+0.0007655, +0.0009349] | 300/300 |
| против `exp-003` standalone LGBM | +0.0008516 | [+0.0006411, +0.0010812] | 300/300 |
| против `exp-005` LGBM blend | -0.0009786 | [-0.0011683, -0.0008193] | 0/300 |

Интервалы отражают row-sampling uncertainty, но не повторное обучение, выбор
конфигурации или Public/Private leaderboard split.

## Бюджет и время

Все пять folds выбрали последнюю, 3000-ю итерацию. Early stopping не сработал:
validation AUC продолжал улучшаться до потолка. Значит, конфигурация не доказала
полное плато CatBoost, хотя честно повторила бюджет `exp-003`.

- CatBoost training и predictions по пяти folds: около `1:27`;
- bootstrap, проверки и запись artifacts: около `2` минут;
- полный запуск: `1:28:58`.

Большой фиксированный preprocessing categorical features амортизировался на
длинном прогоне: увеличение числа деревьев в 15 раз относительно `trial-022` не
увеличило wall time в 15 раз. Тем не менее CatBoost остаётся заметно дороже
LightGBM-линии.

## Submission и Kaggle

```text
file=submission.csv
rows=296302
prediction=mean fold percentile ranks
min=0.000003374935
max=0.999986500260
sha256=6c29b6eb9aca12c2e591550d19d6e8855fe5b2f286c5691a165e1d610c6ef0d1
Kaggle score=0.96711
```

ID полностью совпадают с `sample_submission.csv`, predictions конечны, а
фактический SHA-256 совпадает с `submission.metadata.json`.

Kaggle сохранил локальное ранжирование:

- `exp-006` выше standalone `exp-003` (`0.96681`) на `+0.00030`;
- `exp-006` ниже `exp-005` (`0.96827`) на `-0.00116`.

## Вывод

`exp-006` — новый лучший standalone-кандидат и доказанная победа CatBoost над
одиночным `exp-003`. Однако он не заменяет `exp-005`: LGBM-бленд устойчиво выше
локально и на Kaggle, а обучается существенно дешевле.

Активный поиск возвращается к LightGBM. OOF/test predictions `exp-006` сохранены,
поэтому CatBoost при необходимости можно проверить как post-hoc компонент
межсемейного бленда без повторного обучения, но новая CatBoost-оптимизация сейчас
не приоритетна.
