# exp-004 — High-capacity LightGBM with exact-value categories

- Статус: **completed**
- Дата: 2026-08-15
- Контроли: `exp-002` и `exp-003`
- Модель: конфигурация `exp-002` с exact-value encoding из `exp-003`

## Цель

Проверить, складываются ли два сильных направления: высокая ёмкость и
регуляризация `exp-002` вместе с categorical-копиями точных числовых значений,
давшими основной прирост `exp-003`.

## Конфигурация

```text
folds=5
shuffle=True
seed=2026
n_estimators=8000 maximum per fold
learning_rate=0.02
num_leaves=127
max_depth=-1
min_child_samples=100
reg_alpha=0.0
reg_lambda=5.0
early_stopping_rounds=400
log_every=100
```

Набор исходных и производных признаков совпадает с `exp-003`, включая девять
exact-value categories. Параметры модели полностью перенесены из `exp-002`.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id exp-004 --iterations 8000 --learning-rate 0.02 --num-leaves 127 --max-depth -1 --min-child-samples 100 --reg-lambda 5 --early-stopping-rounds 400 --log-every 100 --drop-features gender stress_level --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep --add-numeric-exact-categories --submission submission.csv
```

## Результат

| Метрика | `exp-002` | `exp-003` | `exp-004` | Delta к `exp-003` |
|---|---:|---:|---:|---:|
| Raw OOF ROC AUC | 0.9640427056 | **0.9650351335** | 0.9645861679 | **-0.0004489656** |
| Fold-rank OOF ROC AUC | 0.9640472606 | **0.9650442019** | 0.9645866927 | **-0.0004575092** |

Для submission выбраны fold ranks. `exp-004` улучшил `exp-002` на всех пяти
folds, но одновременно проиграл `exp-003` на всех пяти folds.

### Fold-wise сравнение

| Fold | `exp-002` | `exp-003` | `exp-004` | Delta к `exp-003` | Best iteration |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9633614731 | **0.9642666519** | 0.9637162324 | -0.0005504194 | 484 |
| 2 | 0.9646593982 | **0.9655261857** | 0.9651113298 | -0.0004148559 | 545 |
| 3 | 0.9649076832 | **0.9659118632** | 0.9655193572 | -0.0003925060 | 529 |
| 4 | 0.9640764927 | **0.9653225469** | 0.9646798288 | -0.0006427182 | 480 |
| 5 | 0.9632312474 | **0.9641937531** | 0.9639067083 | -0.0002870448 | 513 |

Средняя лучшая итерация — `510`, диапазон `480–545`. Все folds завершились по
early stopping задолго до потолка 8000. Высокая ёмкость быстро переобучает
exact-value categories: mean train AUC достиг `0.99324`, несмотря на ранние
лучшие итерации.

### Paired bootstrap

Raw OOF predictions сравнивались на 300 stratified bootstrap-выборках размера
полного test (`296302`).

Против `exp-003`:

```text
point delta:          -0.0004490
paired 95% interval:  [-0.0005516, -0.0003545]
positive repeats:     0/300
```

Против `exp-002`:

```text
point delta:          +0.0005435
paired 95% interval:  [+0.0002768, +0.0007774]
positive repeats:     300/300
```

Интервалы отражают только row-sampling uncertainty, не повторное обучение и не
Public/Private leaderboard split.

## Время

- CV с обучением и predictions: `7:55`;
- полный запуск: `8:05`.

## Submission

```text
file=submission.csv
rows=296302
prediction=mean fold percentile ranks
min=0.000009449818
max=0.999910901715
sha256=aea339f6fb0f04e40ee35f79c6ec98218fdc6c9108aab13f83e409d15eff3c5a
Kaggle score=0.96668
```

Колонки равны `id, addicted_label`, ID полностью совпадают с
`sample_submission.csv`, predictions конечны, фактический SHA-256 совпадает с
`submission.metadata.json`.

## Вывод

Гипотеза частично подтвердилась: exact-value encoding стабильно усилил
конфигурацию `exp-002`, но механическое объединение не превзошло `exp-003`.
`exp-003` остаётся принятым локальным и leaderboard-лидером. `exp-004` полезен
как отрицательный результат и потенциально как отдельный участник ансамбля, но
его одиночный submission локально слабее текущего лидера. Kaggle подтвердил
направление: `0.96668`, на `-0.00013` ниже `exp-003` (`0.96681`).
