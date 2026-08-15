# trial-022 — CatBoost против LightGBM при 200 деревьях

- Статус: **accepted; CatBoost won 5/5 folds and exceeded the target**
- Дата: 2026-08-15
- Внутренний контроль: standalone LightGBM/200 из того же запуска
- Внешний быстрый контроль: `trial-021` LightGBM/250

## Вопрос

Может ли одиночный CatBoost работать на принятом exact-value наборе признаков
не хуже LightGBM при одинаковом бюджете в 200 деревьев, или преимущество
CatBoost проявляется только как небольшая поправка в ансамбле?

## Гипотеза

LightGBM получил сильный прирост от категориальных копий точных числовых
значений, но при этом быстро переобучается: у `trial-019` mean train AUC достиг
`0.99459`. CatBoost строит статистики категорий упорядоченно и может аккуратнее
использовать ту же сетку точных значений. Поэтому проверяется не очередной
blend, а самостоятельная модель.

## Заранее заданный критерий

Основная метрика — raw OOF ROC AUC одиночных моделей. CatBoost считается
достаточно сильным для активной trial-линии, если отстаёт от LightGBM не более
чем на `0.00010`:

```text
CatBoost OOF >= LightGBM OOF - 0.00010
```

Победа blend не учитывается. Submission не создаётся.

## Честная конфигурация

Обе модели получили:

- одни и те же 5 folds `StratifiedKFold(shuffle=True, random_state=2026)`;
- один набор из 22 признаков;
- 10 одинаковых категориальных признаков: `academic_work_impact` и девять
  `_exact`-копий исходных числовых колонок;
- принятые производные признаки `other_screen_time_hours`,
  `screen_social_load`, `high_stress_other_screen_per_sleep`;
- удалённые сырые `gender` и `stress_level`;
- ровно 200 деревьев без early stopping.

Нативные параметры моделей не приравнивались искусственно:

```text
LightGBM: learning_rate=0.10, num_leaves=31, min_child_samples=20,
          cat_smooth=10, reg_alpha=0, reg_lambda=0
CatBoost: learning_rate=0.08, depth=8, l2_leaf_reg=5
```

Для CatBoost это лучшая ранее проверенная конфигурация семейства; параметры
`num_leaves` и `cat_smooth` не имеют прямого эквивалента между библиотеками.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_head_to_head.py --run-id trial-022 --iterations 200 --folds 5 --seed 2026 --log-every 50
```

## Результат

| Метрика | LightGBM/200 | CatBoost/200 | CatBoost − LightGBM |
|---|---:|---:|---:|
| Raw OOF ROC AUC | 0.9642926671 | **0.9650486122** | **+0.0007559451** |
| Fold-rank OOF ROC AUC | 0.9642942152 | **0.9650517759** | **+0.0007575607** |
| Mean train ROC AUC | 0.9821449408 | **0.9716176823** | -0.0105272585 |
| Mean train–validation gap | 0.0178507275 | **0.0065659079** | -0.0112848196 |

CatBoost не просто попал в допуск `0.00010`, а превзошёл LightGBM примерно на
`0.000756`. При этом его train–validation gap значительно меньше.

### Сравнение по folds

| Fold | LightGBM | CatBoost | Delta |
|---:|---:|---:|---:|
| 1 | 0.9634692685 | **0.9641866128** | **+0.0007173443** |
| 2 | 0.9649523453 | **0.9654746154** | **+0.0005222701** |
| 3 | 0.9651148833 | **0.9659197928** | **+0.0008049095** |
| 4 | 0.9645869841 | **0.9653811556** | **+0.0007941715** |
| 5 | 0.9633475852 | **0.9642966953** | **+0.0009491101** |

CatBoost выиграл все пять folds.

### Paired bootstrap raw OOF delta

```text
prediction Pearson correlation:  0.980233
prediction Spearman correlation: 0.968586
mean bootstrap delta:             +0.0007617
standard deviation:               0.0001135
paired 95% interval:              [+0.0005292, +0.0009707]
positive repeats:                 300/300
```

Интервал полностью выше нуля. Он оценивает только row-sampling uncertainty
этой пары predictions и не покрывает смену seed, повторное обучение или
Public/Private split.

## Дополнительный контекст

- LightGBM/200 всего на `+0.0000135` выше принятого LightGBM/250 из
  `trial-021`; это подтверждает наблюдение из live validation, но разница сама
  по себе слишком мала для отдельного вывода.
- CatBoost/200 на `+0.0000135` выше raw OOF дорогого `exp-003`, а его fold-rank
  выше на `+0.0000076`. Это практически равенство, а не доказанная победа над
  `exp-003`, зато вычислительный бюджет — всего 200 деревьев.
- Корреляция предсказаний двух моделей не идеальна, но blend намеренно не
  оценивался: целью был уровень одиночного CatBoost.

## Время

- обучение пяти пар моделей и predictions: около `24:19`;
- paired bootstrap: `0:40`;
- полный запуск: `25:13`.

## Решение

CatBoost/200 **принят как самостоятельная модель активной trial-линии**.
Заранее заданный критерий перевыполнен, победа повторилась на 5/5 folds, а
paired interval полностью положителен. Это также опровергает опасение, что
exact-категории обязательно заставят модель только запоминать train: в этой
конфигурации CatBoost имеет и более высокий validation AUC, и существенно
меньший train–validation gap, чем LightGBM.

Submission не создавался.

Артефакты запуска:

- `artifacts/metrics_trial-022_head_to_head.json`;
- `artifacts/oof_predictions_trial-022.csv`;
- `artifacts/test_predictions_trial-022.csv`;
- `artifacts/feature_importance_trial-022.csv`;
- `artifacts/logs/20260815T145010835423+0300_cv-head-to-head_trial-022.log`.
