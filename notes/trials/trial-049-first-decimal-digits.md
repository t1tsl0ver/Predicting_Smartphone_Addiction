# trial-049 — Первая цифра после запятой

- Статус: **rejected; lost both folds and residual check**
- Дата: 2026-08-21
- Контроль: LightGBM exact/base 60/40 blend из `trial-043`

## Вопрос

Несёт ли первая цифра после запятой у непрерывных часовых колонок отдельный
сигнал синтетического генератора поверх самих числовых и exact-значений?

## Гипотеза

Десятичная решётка может объединять далёкие точные значения, созданные одним
режимом генератора. Обычному числовому дереву такое периодическое объединение
неудобно, а categorical-цифра даёт его напрямую.

## Единственное изменение

Поверх четырёх фич `trial-043` в exact- и base-компоненты добавлены шесть
categorical-признаков первой цифры после запятой:

```text
daily_screen_time_hours_decimal_digit_1
social_media_hours_decimal_digit_1
gaming_hours_decimal_digit_1
work_study_hours_decimal_digit_1
sleep_hours_decimal_digit_1
weekend_screen_time_decimal_digit_1
```

Цифра извлекается без округления последующих знаков. Значение `11.2` даёт
категорию `2`, а `NaN` сохраняется отдельным уровнем. Извлечение сверено с
текстом CSV на всех непустых значениях; расхождений нет. Только 5 значений во
всём train имеют больше двух десятичных знаков, поэтому отдельного заметного
off-grid режима они не образуют.

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
| Base LightGBM | 0.9608080702 | **0.9610209405** | **+0.0002128703** |
| 60/40 blend | **0.9652398017** | 0.9651932220 | **-0.0000465797** |

Blend проиграл оба folds: `-0.0000455000` и `-0.0000477485`. Paired bootstrap
на 100 повторах размера test дал interval
`[-0.0000902068; -0.0000175806]`; положительны только `2/100` повторов.

Exact-компонент не изменился побитово по predictions и дал нулевой gain всем
шести digit-фичам: его categorical-копии точных значений уже содержат этот
сигнал. Base-компонент цифры использовал и вырос, но изменение его ранжирования
ухудшило фиксированную смесь с exact.

## Residual contribution

Без переобучения сравнен фиксированный 50/50 rank blend сильного no-TE
`exp-005` с control/candidate `trial-049`:

| Смесь с `exp-005` | OOF |
|---|---:|
| 50% `exp-005` + 50% `trial-043` | **0.9665859581** |
| 50% `exp-005` + 50% `trial-049` | 0.9665724573 |

Delta `-0.0000135008`; оба screening folds отрицательны. Bootstrap на 300
повторах пересекает ноль: `[-0.0000329574; +0.0000044187]`. Значит, первая
цифра не исправляет остаточные ошибки сильного no-TE ансамбля.

## Вывод

Первая цифра после запятой **отклонена**. Небольшой standalone-прирост base не
перенёсся ни в основной exact/base blend, ни в residual-проверку. В активный
набор признаки не добавляются; 5-fold подтверждение не требуется.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-049 --candidate screening_winners_plus_first_decimal_digits --control-oof artifacts\oof_predictions_trial-043.csv --control-metrics artifacts\metrics_trial-043_feature.json
```

Артефакты: `artifacts/metrics_trial-049_feature.json`, OOF, importance и лог
`20260821T160721593438+0300_cv-feature-trial_trial-049.log`.
