# exp-003 — Exact-value LightGBM with a low learning rate

- Статус: **completed**
- Дата: 2026-08-15
- Контроли: `trial-019` и `exp-002`
- Модель: чистый LightGBM с numeric exact-value categories

## Цель

Перенести принятый exact-value encoding из `trial-019` в полноценный режим
обучения с меньшим learning rate, individual best iteration для каждого fold и
созданием нового канонического Kaggle submission.

## Принятые признаки

- исходные признаки без сырых `gender` и `stress_level`;
- `academic_work_impact` сохранён после отрицательного `trial-018`;
- `other_screen_time_hours` из `trial-010`;
- `screen_social_load` из `trial-012`;
- `high_stress_other_screen_per_sleep` из `trial-014`;
- девять categorical-копий точных числовых значений из `trial-019`;
- исходные числовые версии всех девяти колонок также сохранены.

## Конфигурация

```text
folds=5
shuffle=True
seed=2026
n_estimators=3000 maximum per fold
learning_rate=0.01
num_leaves=31
max_depth=-1
min_child_samples=20
reg_alpha=0.0
reg_lambda=0.0
early_stopping_rounds=400
log_every=100
```

`early_stopping_rounds=400` означает patience, а не ограничение модели 400
деревьями. Fold мог построить до 3000 деревьев и завершался, если validation AUC
не обновлял максимум 400 раундов подряд. Predictions делались на лучшей
сохранённой итерации.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id exp-003 --iterations 3000 --learning-rate 0.01 --early-stopping-rounds 400 --log-every 100 --drop-features gender stress_level --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep --add-numeric-exact-categories --submission submission.csv
```

## Результат

| Метрика | `trial-019` | `exp-002` | `exp-003` | Delta к `exp-002` |
|---|---:|---:|---:|---:|
| Raw OOF ROC AUC | 0.9640249964 | 0.9640427056 | **0.9650351335** | **+0.0009924279** |
| Fold-rank OOF ROC AUC | 0.9640257010 | 0.9640472606 | **0.9650442019** | **+0.0009969413** |

Для submission выбраны fold ranks. `exp-003` выиграл у прежнего локального
лидера `exp-002` на всех пяти folds.

### Fold-wise сравнение с `exp-002`

| Fold | `exp-002` | `exp-003` | Delta | Best iteration `exp-003` |
|---:|---:|---:|---:|---:|
| 1 | 0.9633614731 | **0.9642666519** | **+0.0009051788** | 2986 |
| 2 | 0.9646593982 | **0.9655261857** | **+0.0008667874** | 2631 |
| 3 | 0.9649076832 | **0.9659118632** | **+0.0010041800** | 2298 |
| 4 | 0.9640764927 | **0.9653225469** | **+0.0012460542** | 2999 |
| 5 | 0.9632312474 | **0.9641937531** | **+0.0009625057** | 2482 |

Средняя лучшая итерация — `2679`, диапазон `2298–2999`. Early stopping завершил
обучение до потолка на третьем и пятом folds. Первый и четвёртый folds нашли
лучшие точки почти у 3000, поэтому текущего бюджета хватило для сильного
результата, но он не гарантирует полностью найденное плато каждого fold.

### Paired bootstrap против `exp-002`

Raw OOF predictions сравнивались на одних строках через 300 stratified bootstrap
выборок размера полного test (`296302`):

```text
prediction Pearson correlation:  0.967613
prediction Spearman correlation: 0.973772
point delta:                      +0.0009924
mean bootstrap delta:             +0.0009823
standard deviation:               0.0001260
paired 95% interval:              [+0.0007320, +0.0011896]
positive repeats:                 300/300
```

Интервал оценивает только row-sampling uncertainty этой пары. Он не моделирует
повторное обучение, distribution shift или Public/Private leaderboard split.

## Время

- CV с обучением и predictions: `20:53`;
- полный запуск с подготовкой и сохранением: `21:03`.

Большая часть времени после обучения fold уходила на predictions моделей с
2300–3000 деревьями по train, validation и test.

## Submission

```text
file=submission.csv
rows=296302
prediction=mean fold percentile ranks
min=0.000004724909
max=0.999994600104
sha256=c4f72dd11b12d50833772da886986ba23516f3fee6471901ee59856bb271e960
Kaggle score=0.96681
```

Колонки равны `id, addicted_label`, ID полностью совпадают с
`sample_submission.csv`, predictions конечны, а фактический SHA-256 совпадает с
`submission.metadata.json`.

## Вывод

Маленький learning rate и individual best iteration раскрыли exact-value
encoding сильнее фиксированных 500 деревьев. `exp-003` — новый локальный лидер:
fold-rank OOF `0.9650442`, прирост почти `+0.0010` к `exp-002`, победа 5/5 folds
и полностью положительный paired row bootstrap. Kaggle подтвердил улучшение:
score `0.96681`, что на `+0.00215` выше `exp-001` (`0.96466`).
