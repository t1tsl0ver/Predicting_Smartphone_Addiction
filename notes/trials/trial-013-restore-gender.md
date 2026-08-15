# trial-013 — Restore gender

- Статус: **rejected**
- Дата: 2026-08-14
- Контроль: `trial-012`

## Вопрос

Добавляет ли исходный категориальный `gender` полезный сигнал поверх принятого
`screen_social_load` и остальных фич `trial-012`?

## Гипотеза

На строках без пропусков в `gender`, `daily_screen_time_hours` и
`social_media_hours` доля положительного target у Male (`0.7231`) немного выше,
чем у Female (`0.7039`) и Other (`0.7008`). При этом сила связи screen/social
product почти одинакова во всех группах.

Возврат `gender` может позволить LightGBM учесть небольшой групповой сдвиг и при
необходимости построить отдельные ветви поверх общего `screen_social_load`.
Ранее `gender` удалялся одновременно со `stress_level`, поэтому его собственный
вклад на текущем наборе ещё не измерялся изолированно.

## Единственное основное изменение

Возвращается исходная категориальная колонка `gender`. `stress_level` остаётся
удалённым. Новые gender-specific product-фичи не создаются.

## Что остаётся неизменным

```text
model=LGBMClassifier
n_estimators=500
learning_rate=0.1
num_leaves=31
min_child_samples=20
reg_alpha=0.0
reg_lambda=0.0
folds=5
shuffle=True
random_state=2026
drop_features=[stress_level]
add_other_screen_time=True
add_screen_social_load=True
add_screen_awake_share=False
```

## План проверки

- использовать те же folds, что в `trial-012`;
- сравнить общий raw OOF AUC и fold-wise deltas;
- принять возврат `gender` при положительном общем delta и победе минимум на
  3/5 folds.

## Команда

```powershell
.\.venv\Scripts\python.exe src\cv_lightgbm.py --run-id trial-013 --drop-features stress_level --add-other-screen-time --add-screen-social-load --submission submission_trial-013.csv
```

## Результат

| Метрика | `trial-012` | `trial-013` | Delta |
|---|---:|---:|---:|
| Raw OOF ROC AUC | **0.9620845212** | 0.9619788289 | **-0.0001056923** |
| Fold-rank OOF ROC AUC | **0.9620866536** | 0.9619813588 | **-0.0001052948** |

Для submission выбраны fold ranks. CV занял `111.93` секунды.

### Сравнение по folds

| Fold | `trial-012` | `trial-013` | Delta |
|---:|---:|---:|---:|
| 1 | 0.9605890951 | **0.9614495077** | **+0.0008604126** |
| 2 | 0.9622416118 | **0.9630338485** | **+0.0007922368** |
| 3 | 0.9632267364 | **0.9633070915** | **+0.0000803550** |
| 4 | **0.9627634979** | 0.9616385075 | -0.0011249904 |
| 5 | **0.9616116948** | 0.9604746921 | -0.0011370027 |

`gender` выиграл на 3/5 folds, но два проигрыша оказались крупнее трёх побед.
Общий delta отрицательный, поэтому критерий принятия не выполнен.

Средний gain importance `gender` равен всего `290.83`, тогда как importance
содержательных числовых фич измеряется миллионами. Модель практически не
использует возвращённую колонку.

## Решение

Возврат `gender` **отклонён**. Активным контролем остаётся `trial-012`, где
удалены и `gender`, и `stress_level`.

Для воспроизводимости сохранён `submission_trial-013.csv`. SHA-256:
`47c9007f233e6d7928fa636195694bca72ef214f51bfdb0732f69660ecb0cc4c`.
Загружать его вместо submission от `trial-012` не рекомендуется.
