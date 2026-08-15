# trial-019 — Exact-value categories for numeric features

- Статус: **accepted; positive paired OOF and won 5/5 folds**
- Дата: 2026-08-15
- Контроль: `trial-014`

## Вопрос

Содержит ли дискретная сетка значений синтетического генератора дополнительный
сигнал, который LightGBM не извлекает из обычного числового порядка признаков?

## Гипотеза

Почти каждое точное значение test уже встречается в train: покрытие test-строк
для отдельных числовых колонок составляет `99.99–100%`. При 691 369 строках
число уникальных значений сравнительно мало:

| Признак | Уникальных значений train |
|---|---:|
| `age` | 18 |
| `daily_screen_time_hours` | 1389 |
| `social_media_hours` | 721 |
| `gaming_hours` | 401 |
| `work_study_hours` | 600 |
| `sleep_hours` | 451 |
| `notifications_per_day` | 231 |
| `app_opens_per_day` | 166 |
| `weekend_screen_time` | 1437 |

Это похоже на округлённую решётку генератора. Числовое дерево использует порядок
и пороги, но не обязано удобно выделять отдельные точки решётки. Категориальная
копия позволяет искать группы точных значений без потери исходного числового
представления.

## Единственное основное изменение

Для девяти исходных числовых колонок добавлены категориальные копии с суффиксом
`_exact`. Обычные числовые версии сохранены. Производные признаки не кодируются
как exact categories.

Пример:

```text
gaming_hours=1.37
gaming_hours_exact="1.37"
```

Пропуски сохраняются и затем обрабатываются общей категорией `__MISSING__`.

## Что остаётся неизменным

```text
model=LGBMClassifier
n_estimators=500
learning_rate=0.1
num_leaves=31
max_depth=-1
min_child_samples=20
reg_alpha=0.0
reg_lambda=0.0
early_stopping_rounds=0
folds=5
shuffle=True
random_state=2026
drop_features=[gender, stress_level]
add_other_screen_time=True
add_screen_social_load=True
add_high_stress_other_screen_per_sleep=True
```

## План проверки

- использовать те же folds, что в `trial-014`;
- сравнить raw и fold-rank OOF ROC AUC и каждый fold;
- сохранить исходные числовые колонки, чтобы не смешивать добавление
  представления с их удалением;
- выполнить paired stratified bootstrap разницы raw OOF на выборках размера
  полного test (`296302` строк);
- считать bootstrap только оценкой row-sampling uncertainty, а не гарантией
  повторного обучения или Private LB;
- не создавать submission до положительного локального результата.

## Команды

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id trial-019 --drop-features gender stress_level --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep --add-numeric-exact-categories

.\.venv\Scripts\python.exe -u src\compare_oof.py --control artifacts\oof_predictions_trial-014.csv --candidate artifacts\oof_predictions_trial-019.csv --prediction-column lightgbm_probability --repeats 300 --sample-size 296302 --seed 2026 --output artifacts\comparison_trial-014_vs_trial-019.json
```

## Результаты

| Метрика | `trial-014` | `trial-019` | Delta |
|---|---:|---:|---:|
| Raw OOF ROC AUC | 0.9622430857 | **0.9640249964** | **+0.0017819107** |
| Fold-rank OOF ROC AUC | 0.9622452251 | **0.9640257010** | **+0.0017804759** |

Fold ranks оказались выше raw probabilities только на `0.0000007`. CV занял
`300.28` секунды, примерно в `3.21x` дольше контроля; полный запуск с подготовкой
категорий и сохранением занял `5:29`.

### Сравнение по folds

| Fold | `trial-014` | `trial-019` | Delta |
|---:|---:|---:|---:|
| 1 | 0.9618137470 | **0.9631629673** | **+0.0013492203** |
| 2 | 0.9628245113 | **0.9647113599** | **+0.0018868485** |
| 3 | 0.9632180879 | **0.9647552505** | **+0.0015371626** |
| 4 | 0.9622383879 | **0.9643327627** | **+0.0020943748** |
| 5 | 0.9611308178 | **0.9631661562** | **+0.0020353384** |

Изменение выиграло все пять folds.

### Paired bootstrap raw OOF delta

```text
prediction Pearson correlation:  0.959478
prediction Spearman correlation: 0.968084
repeats:                          300
sample size per repeat:           296302
mean delta:                       +0.0017729
standard deviation:               0.0001583
paired 95% interval:              [+0.0014480, +0.0020813]
positive repeats:                 300/300
```

Интервал не включает ноль и подтверждает, что прирост значительно выше
row-sampling noise floor этой конкретной пары predictions. Он не учитывает
вариативность обучения, выбор Public/Private строк или возможный distribution
shift.

### Сравнение с `exp-002`

`trial-019` почти сравнялся с дорогой моделью: raw OOF ниже всего на
`0.0000177`, fold-rank OOF — на `0.0000216`. Это не является честной победой над
`exp-002`, но показывает, что новое представление дало основной прирост без
увеличения числа folds или бюджета до тысяч деревьев.

## Наблюдения

- Exact-копии получили большую importance; особенно сильны
  `notifications_per_day_exact`, `app_opens_per_day_exact`, exact-копии daily и
  weekend screen time.
- Одновременно сохранилась высокая importance обычных числовых daily, social и
  weekend screen time. Оба представления действительно используются.
- Mean train AUC достиг `0.99459` против validation около `0.964`, поэтому риск
  переобучения заметно выше контроля.
- В сохранённом логе validation AUC на отметке 200 итераций выше финальной
  500-й на всех folds. Точная лучшая итерация не записана, поскольку early
  stopping был отключён. Это отдельная гипотеза для следующего trial.

## Вывод

Exact-value categorical encoding **принят**: общий прирост крупный, изменение
выиграло 5/5 folds, а paired row bootstrap полностью положителен. `trial-019`
становится активным контролем быстрой линии. Submission не создавался.

Артефакты запуска:

- `artifacts/metrics_trial-019_cv.json`;
- `artifacts/oof_predictions_trial-019.csv`;
- `artifacts/test_predictions_trial-019.csv`;
- `artifacts/feature_importance_trial-019.csv`;
- `artifacts/comparison_trial-014_vs_trial-019.json`;
- `artifacts/logs/20260815T115441712428+0300_cv-lightgbm_trial-019.log`.
