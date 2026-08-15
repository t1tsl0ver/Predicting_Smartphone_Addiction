# exp-005 — Conservative rank blend of exp-003 and exp-002

- Статус: **completed locally; Kaggle score pending**
- Дата: 2026-08-15
- Компоненты: `exp-003` и `exp-002`
- Метод: фиксированный weighted rank blend без переобучения моделей

## Цель

Объединить две модели, использующие разные представления и параметры:

- `exp-003` — exact-value categories, 31 leaf, LR `0.01`;
- `exp-002` — без exact-value categories, 127 leaves, LR `0.02`, L2 `5`.

Их fold-rank predictions имеют корреляцию около `0.974`, поэтому более слабый
одиночный `exp-002` может исправлять часть ошибок `exp-003`.

## Выбор весов

Предварительный OOF sweep дал широкий максимум:

| Вес `exp-003` | Вес `exp-002` | OOF AUC |
|---:|---:|---:|
| 70% | 30% | 0.9666911 |
| 60% | 40% | **0.9668744** |
| 55% | 45% | 0.9668965 |
| 50% | 50% | 0.9668719 |

Формальный максимум 55/45 выше 60/40 только на `0.0000221`. Выбран более
консервативный вес 60/40, оставляющий большую долю модели `exp-003`, которая уже
получила Kaggle score `0.96681`. Вес не подгонялся по leaderboard.

## Реализация

Добавлен `src/blend_cv.py`. Скрипт:

- проверяет совпадение OOF rows, targets и folds;
- проверяет совпадение test ID и sample submission;
- смешивает fold percentile ranks и test CV ranks;
- считает общий и fold-wise OOF AUC;
- выполняет paired stratified bootstrap против primary-модели;
- сохраняет OOF/test predictions, метрики, submission и метаданные.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\blend_cv.py --primary-oof artifacts\oof_predictions_exp-003.csv --primary-test artifacts\test_predictions_exp-003.csv --primary-name exp-003 --primary-weight 0.60 --secondary-oof artifacts\oof_predictions_exp-002.csv --secondary-test artifacts\test_predictions_exp-002.csv --secondary-name exp-002 --run-id exp-005 --submission submission.csv
```

## Результат

| Модель | OOF ROC AUC | Delta к `exp-003` |
|---|---:|---:|
| `exp-002` | 0.9640472606 | -0.0009969413 |
| `exp-003` | 0.9650442019 | — |
| `exp-005` 60/40 rank blend | **0.9668743684** | **+0.0018301665** |

### Сравнение по folds

| Fold | `exp-003` | `exp-005` | Delta |
|---:|---:|---:|---:|
| 1 | 0.9642666519 | **0.9660595333** | **+0.0017928815** |
| 2 | 0.9655261857 | **0.9674243147** | **+0.0018981290** |
| 3 | 0.9659118632 | **0.9677176890** | **+0.0018058259** |
| 4 | 0.9653225469 | **0.9671165472** | **+0.0017940002** |
| 5 | 0.9641937531 | **0.9660509997** | **+0.0018572466** |

Бленд выиграл все пять folds, а размер прироста очень стабилен.

### Paired bootstrap против `exp-003`

```text
component Pearson correlation:   0.974082
component Spearman correlation:  0.974082
point delta:                      +0.0018302
mean bootstrap delta:             +0.0018344
standard deviation:               0.0000516
paired 95% interval:              [+0.0017422, +0.0019386]
positive repeats:                 300/300
```

Интервал отражает только row-sampling uncertainty. Он не учитывает выбор веса,
повторное обучение или Public/Private leaderboard split.

## Submission

```text
file=submission.csv
rows=296302
prediction=0.60 * exp-003 CV rank + 0.40 * exp-002 CV rank
min=0.000012014769
max=0.999890517108
sha256=546dddd18ae79e7220266e07801c042aaba6faf37ef57dc42f6f46d183e6ce32
Kaggle score=pending
```

Колонки равны `id, addicted_label`, ID полностью совпадают с
`sample_submission.csv`, predictions конечны, а SHA-256 совпадает с
`submission.metadata.json`.

## Вывод

`exp-005` становится новым локальным лидером. Прирост существенно больше
наблюдаемого noise floor и согласован на всех folds. Главный оставшийся риск —
перенос подобранного OOF-веса на Public/Private leaderboard; окончательное
решение принимается после Kaggle score.
