# trial-021 — Stop exact-value LightGBM at 250 trees

- Статус: **accepted; positive paired OOF and won 5/5 folds**
- Дата: 2026-08-15
- Контроль: `trial-019`

## Вопрос

Улучшит ли раннее завершение быстрого exact-value LightGBM обобщение, если его
validation AUC на всех folds достигает максимума заметно раньше 500 деревьев?

## Гипотеза

В `trial-019` mean train AUC достиг `0.99459`, а validation после примерно
200–250 итераций начинал снижаться. `trial-020` показал, что categorical
smoothing уменьшает train AUC, но не улучшает OOF. Более прямой способ убрать
лишнюю подгонку — сократить boosting budget без изменения признаков и
регуляризации.

## Единственное основное изменение

```text
n_estimators: 500 -> 250
```

Остальная конфигурация совпадает с принятым `trial-019`, включая дефолтный
`cat_smooth=10`, 31 leaf, learning rate `0.1`, те же признаки, folds и seeds.
Early stopping не включался, чтобы сравнивать один фиксированный бюджет с
другим.

## Команды

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id trial-021 --iterations 250 --drop-features gender stress_level --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep --add-numeric-exact-categories

.\.venv\Scripts\python.exe -u src\compare_oof.py --control artifacts\oof_predictions_trial-019.csv --candidate artifacts\oof_predictions_trial-021.csv --prediction-column lightgbm_probability --repeats 300 --sample-size 296302 --seed 2026 --output artifacts\comparison_trial-019_vs_trial-021.json
```

## Результат

| Метрика | `trial-019` | `trial-021` | Delta |
|---|---:|---:|---:|
| Raw OOF ROC AUC | 0.9640249964 | **0.9642792125** | **+0.0002542161** |
| Fold-rank OOF ROC AUC | 0.9640257010 | **0.9642794731** | **+0.0002537721** |

Для внутренних test predictions выбран fold-rank вариант. Submission не
создавался.

### Сравнение по folds

| Fold | `trial-019` | `trial-021` | Delta |
|---:|---:|---:|---:|
| 1 | 0.9631629673 | **0.9634245205** | **+0.0002615532** |
| 2 | 0.9647113599 | **0.9649406455** | **+0.0002292856** |
| 3 | 0.9647552505 | **0.9650957949** | **+0.0003405444** |
| 4 | 0.9643327627 | **0.9646172569** | **+0.0002844942** |
| 5 | 0.9631661562 | **0.9633191376** | **+0.0001529814** |

Изменение выиграло все пять folds. Mean train AUC снизился с `0.99459` до
`0.98517`, а mean train–validation gap — примерно с `0.03057` до `0.02089`.

### Paired bootstrap raw OOF delta

```text
prediction Pearson correlation:  0.996745
prediction Spearman correlation: 0.997051
point delta:                      +0.0002542
mean bootstrap delta:             +0.0002505
standard deviation:               0.0000437
paired 95% interval:              [+0.0001639, +0.0003373]
positive repeats:                 300/300
```

Интервал полностью выше нуля и отражает только row-sampling uncertainty этой
пары predictions.

## Время

- CV: `1:13`;
- полный запуск: `1:23`.

## Решение

250 деревьев **приняты** и становятся новым быстрым контролем. Результат
показывает, что 500 деревьев при learning rate `0.1` переобучали exact-value
модель. На отметке 200 итераций live validation AUC был немного выше 250-й на
четырёх folds из пяти, поэтому 200 деревьев остаются отдельной проверяемой
гипотезой, а не автоматически принятым значением.

Артефакты запуска:

- `artifacts/metrics_trial-021_cv.json`;
- `artifacts/oof_predictions_trial-021.csv`;
- `artifacts/test_predictions_trial-021.csv`;
- `artifacts/feature_importance_trial-021.csv`;
- `artifacts/comparison_trial-019_vs_trial-021.json`;
- `artifacts/logs/20260815T141120202672+0300_cv-lightgbm_trial-021.log`.
