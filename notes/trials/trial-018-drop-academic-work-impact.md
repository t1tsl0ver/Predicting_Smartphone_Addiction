# trial-018 — Drop academic/work impact

- Статус: **rejected; raw OOF decreased despite 3/5 fold wins**
- Дата: 2026-08-15
- Контроль: `trial-014`

## Вопрос

Помогает ли модели слабый категориальный признак `academic_work_impact`, или его
удаление уменьшит шум и улучшит обобщение?

## Гипотеза

Одномерная связь категории с target очень мала: доля положительного класса равна
`0.707883` для `Yes`, `0.711029` для `No` и `0.709456` для пропусков. В
`trial-014` признак также занимал последнее место по mean gain importance
(`431818.85`, тогда как у следующего признака было `3356307.79`). Поэтому
LightGBM мог тратить часть разбиений на слабый или случайный сигнал.

При этом низкая importance не доказывает бесполезность: категория могла помогать
в отдельных взаимодействиях или через собственный паттерн пропусков.

## Единственное основное изменение

К списку удалённых сырых признаков `gender` и `stress_level` добавлен
`academic_work_impact`. Числовой `work_study_hours` и все принятые производные
признаки сохранены.

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
add_other_screen_time=True
add_screen_social_load=True
add_high_stress_other_screen_per_sleep=True
```

## План проверки

- использовать те же folds, что в `trial-014`;
- сравнить raw и fold-rank OOF ROC AUC;
- сравнить validation AUC на каждом fold;
- принять удаление только при положительном raw OOF delta и победе минимум на
  3/5 folds;
- не создавать submission до положительного локального результата.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id trial-018 --drop-features gender stress_level academic_work_impact --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep
```

## Результаты

| Метрика | `trial-014` | `trial-018` | Delta |
|---|---:|---:|---:|
| Raw OOF ROC AUC | **0.9622430857** | 0.9620822904 | **-0.0001607952** |
| Fold-rank OOF ROC AUC | **0.9622452251** | 0.9620816383 | **-0.0001635868** |

Raw probabilities оказались немного лучше fold ranks, однако оба варианта ниже
контроля. CV занял `91.02` секунды; полный запуск с подготовкой и сохранением —
около `1:35`.

### Сравнение по folds

| Fold | `trial-014` | `trial-018` | Delta |
|---:|---:|---:|---:|
| 1 | **0.9618137470** | 0.9610783537 | -0.0007353933 |
| 2 | 0.9628245113 | **0.9629701687** | +0.0001456574 |
| 3 | **0.9632180879** | 0.9627522076 | -0.0004658803 |
| 4 | 0.9622383879 | **0.9622993098** | +0.0000609219 |
| 5 | 0.9611308178 | **0.9613088441** | +0.0001780263 |

## Наблюдения

Удаление выиграло 3/5 folds, но все три улучшения малы. Два проигрыша, особенно
`-0.0007354` на первом fold, полностью их перевесили. Это ещё один пример того,
что число fold-побед нельзя рассматривать отдельно от размера изменений и
общего OOF.

После удаления в модели не осталось сырых категориальных колонок. Полученный
негативный delta показывает, что почти нулевая gain importance не равна нулевой
предсказательной ценности. По этому trial нельзя определить, помогала сама
категория или информация о её пропусках.

## Вывод

Удаление `academic_work_impact` **отклонено**: обязательное условие положительного
raw OOF delta не выполнено. Признак сохраняется, `trial-014` остаётся активным
быстрым контролем. Submission не создавался.

Артефакты запуска:

- `artifacts/metrics_trial-018_cv.json`;
- `artifacts/oof_predictions_trial-018.csv`;
- `artifacts/test_predictions_trial-018.csv`;
- `artifacts/feature_importance_trial-018.csv`;
- `artifacts/logs/20260815T113708560797+0300_cv-lightgbm_trial-018.log`.
