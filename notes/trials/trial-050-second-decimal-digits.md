# trial-050 — Вторая цифра после запятой

- Статус: **accepted for screening; 5-fold confirmation required**
- Дата: 2026-08-21
- Контроль: LightGBM exact/base 60/40 blend из `trial-043`

## Вопрос

Несёт ли вторая цифра после запятой у непрерывных часовых колонок отдельный
сигнал синтетического генератора поверх самих числовых и exact-значений?

## Гипотеза

Как и первая цифра, вторая задаёт периодическую группу точных значений, которую
обычным числовым splits трудно представить. Она проверяется независимо от
отклонённой первой цифры, чтобы два эффекта не маскировали друг друга.

## Единственное изменение

Поверх четырёх фич `trial-043` в exact- и base-компоненты добавлены шесть
categorical-признаков второй цифры после запятой:

```text
daily_screen_time_hours_decimal_digit_2
social_media_hours_decimal_digit_2
gaming_hours_decimal_digit_2
work_study_hours_decimal_digit_2
sleep_hours_decimal_digit_2
weekend_screen_time_decimal_digit_2
```

Цифра извлекается без округления последующих знаков: `11.2` даёт `0`, а
`4.549` даёт `4`. `NaN` сохраняется отдельным уровнем. Извлечение сверено с
текстом CSV на всех непустых значениях; расхождений нет.

## Что остаётся неизменным

- 2 stratified folds, seed `2026`;
- 200 деревьев, learning rate `0.10`;
- exact/base fold-rank blend `60/40`;
- четыре обычные фичи `trial-043`, без target encoding;
- cached OOF `trial-043` как контроль.

Критерий screening: положительная общая delta, победа 2/2 folds и полностью
положительный paired-bootstrap interval.

## Результаты

| Вариант | Control OOF | Candidate OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.9637208762 | 0.9637208762 | 0.0000000000 |
| Base LightGBM | 0.9608080702 | **0.9616847035** | **+0.0008766333** |
| 60/40 blend | 0.9652398017 | **0.9652975730** | **+0.0000577713** |

Blend выиграл оба folds: `+0.0000790510` и `+0.0000360890`. Paired bootstrap
на 100 повторах размера test полностью положителен:
`[+0.0000124392; +0.0000932940]`, положительны `100/100` повторов.

Exact-компонент снова сохранил те же fold ranks; максимальное отличие raw
probability составило `1.2e-15`. Все digit-фичи получили нулевой gain:
exact-категории уже кодируют полное значение. Base-компонент получил заметный
прирост. Наибольший mean gain среди новых фич получили вторая цифра
`weekend_screen_time` (`6697.55`), `daily_screen_time_hours` (`5116.53`) и
`sleep_hours` (`2104.53`). Gain объясняет использование, но не причинность.

## Residual contribution

Без переобучения сравнен фиксированный 50/50 rank blend сильного no-TE
`exp-005` с control/candidate:

| Смесь с `exp-005` | OOF |
|---|---:|
| 50% `exp-005` + 50% `trial-043` | 0.9665859581 |
| 50% `exp-005` + 50% `trial-050` | **0.9666371484** |

Delta `+0.0000511902`; оба screening folds положительны: `+0.0000635685` и
`+0.0000387724`. Bootstrap на 300 повторах полностью выше нуля:
`[+0.0000310726; +0.0000712926]`, положительны `300/300` повторов. Spearman
между `exp-005` и candidate равен `0.9934489`: predictions близки, но новый
сигнал не исчезает рядом с сильным no-TE ансамблем.

## Вывод

Вторая цифра после запятой **принята для screening**. Результат противоположен
заявлению разобранного стороннего notebook, поэтому особенно важно не переносить
его сразу в финальную модель. Delta мала, получена после множества предыдущих
feature-проверок и пока основана только на двух folds. Перед включением нужен
отдельно разрешённый 5-fold no-TE эксперимент; первую цифру в него не переносить.

Результат показывает добавочный локальный сигнал для base-LightGBM, но сам по
себе не доказывает, что цифра является причинным fingerprint генератора.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-050 --candidate screening_winners_plus_second_decimal_digits --control-oof artifacts\oof_predictions_trial-043.csv --control-metrics artifacts\metrics_trial-043_feature.json
```

Артефакты: `artifacts/metrics_trial-050_feature.json`, OOF, importance и лог
`20260821T160842401580+0300_cv-feature-trial_trial-050.log`.
