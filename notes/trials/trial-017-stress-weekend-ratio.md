# trial-017 — Raw stress + weekend/daily ratio

- Статус: **rejected**
- Дата: 2026-08-14
- Контроль: `trial-014`

## Гипотеза

Проверить пакет изменений, приближающий чистый LightGBM к конфигурации с
digital-признаками:

```text
weekend_to_daily_screen_ratio =
    weekend_screen_time / daily_screen_time_hours
```

Отношение отделяет абсолютную длительность использования смартфона от изменения
поведения на выходных. Значение выше единицы означает, что на выходных экранное
время растёт относительно обычного дня. Исходный `stress_level` возвращается,
чтобы LightGBM мог самостоятельно использовать все его категории и строить
взаимодействия, а не получать только High-индикатор составной фичи.

## Изменения относительно контроля

- `stress_level` возвращается в признаки;
- добавляется `weekend_to_daily_screen_ratio`;
- `gender` остаётся исключённым;
- все три принятые FE из `trial-014` сохраняются.

Если знаменатель отсутствует или не положителен, ratio остаётся `NaN`, который
LightGBM обрабатывает нативно.

Эксперимент проверяет пакет из двух изменений, поэтому по его результату нельзя
отдельно определить вклад исходного stress и нового ratio.

## Неизменная конфигурация

```text
model=LGBMClassifier
n_estimators=500
learning_rate=0.1
num_leaves=31
max_depth=-1
min_child_samples=20
reg_alpha=0.0
reg_lambda=0.0
folds=5
shuffle=True
random_state=2026
drop_features=[gender]
add_other_screen_time=True
add_screen_social_load=True
add_high_stress_other_screen_per_sleep=True
submission=False
```

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id trial-017 --iterations 500 --learning-rate 0.1 --drop-features gender --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep --add-weekend-to-daily-screen-ratio
```

Первый запуск с `learning_rate=0.05` был остановлен во время пятого fold по
решению участника и не считается результатом trial.

## Критерий

Сравниваем с `trial-014` общий raw и fold-rank OOF ROC AUC, а также изменения на
каждом из пяти folds. Submission не создаётся.

## Результат

| Метрика | `trial-014` | `trial-017` | Delta |
|---|---:|---:|---:|
| Raw OOF ROC AUC | **0.9622430857** | 0.9619782259 | **-0.0002648598** |
| Fold-rank OOF ROC AUC | **0.9622452251** | 0.9619827462 | **-0.0002624789** |

CV занял `1:55`. Для predictions был выбран fold-rank вариант, но submission не
создавался.

### Сравнение по folds

| Fold | `trial-014` | `trial-017` | Delta |
|---:|---:|---:|---:|
| 1 | 0.9618137470 | **0.9619154711** | **+0.0001017240** |
| 2 | 0.9628245113 | **0.9632696901** | **+0.0004451788** |
| 3 | 0.9632180879 | **0.9632286931** | **+0.0000106052** |
| 4 | **0.9622383879** | 0.9601042654 | **-0.0021341225** |
| 5 | 0.9611308178 | **0.9613946629** | **+0.0002638451** |

Новый пакет выиграл 4/5 folds, но сильный провал четвёртого fold полностью
перекрыл четыре небольших улучшения. `weekend_to_daily_screen_ratio` занял 9-е
место из 15 по mean gain (`16.76 млн`). Возвращённый `stress_level` оказался
последним с gain всего `201`, то есть модель практически его не использовала.

## Решение

Пакет **отклонён**, потому что общий OOF ниже контроля. `trial-014` остаётся
активной конфигурацией быстрых trial. По этому запуску нельзя строго разделить
эффект ratio и stress, однако почти нулевая importance сырого stress позволяет
предположить, что основное изменение поведения модели связано с ratio.
