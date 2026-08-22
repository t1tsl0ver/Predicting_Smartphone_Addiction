# trial-052 — Вторые десятичные цифры на лёгком 5-fold baseline

- Статус: **weak positive; not adopted**
- Дата: 2026-08-22
- Контроль: принятый лёгкий пакет `trial-051`

## Вопрос

Сохраняется ли слабый сигнал второй цифры после запятой из 2-fold `trial-050`,
если добавить эти признаки к текущему лёгкому 5-fold baseline?

## Гипотеза

Вторая десятичная цифра может отражать дискретную решётку синтетического
генератора, которую base-LightGBM трудно восстановить обычными числовыми splits.
При этом она не использует target и не является target encoding. Риск — старый
прирост `trial-050` был мал и мог оказаться особенностью двух folds.

## Единственное изменение

Поверх пяти фич `trial-051` в exact- и base-компоненты добавлены шесть
categorical-признаков:

```text
daily_screen_time_hours_decimal_digit_2
social_media_hours_decimal_digit_2
gaming_hours_decimal_digit_2
work_study_hours_decimal_digit_2
sleep_hours_decimal_digit_2
weekend_screen_time_decimal_digit_2
```

Цифра извлекается без округления последующих знаков. Пропуск остаётся отдельным
categorical-уровнем. Target, frequency lookup и новые агрегаты не используются.

## Неизменная конфигурация

- 5 stratified folds, shuffle, seed `2026`;
- 200 деревьев, learning rate `0.10`, без early stopping;
- exact/base fold-rank blend `60/40`;
- контрольные OOF `trial-051` на тех же строках и folds;
- без тяжёлых моделей и без создания submission.

## Результаты

| Вариант | `trial-051` | Кандидат | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.9651207118 | 0.9651207118 | 0.0000000000 |
| Base LightGBM | 0.9618772645 | **0.9624128666** | **+0.0005356021** |
| 60/40 blend | 0.9660227661 | **0.9660553084** | **+0.0000325423** |

Exact predictions полностью совпали с контролем: Spearman `1.0`, все шесть
новых фич получили нулевой gain. Полные exact-категории уже содержат эту
информацию. В base прирост заметный, но фиксированный blend разбавил его сильным
неизменившимся exact-компонентом.

| Fold | Delta blend |
|---:|---:|
| 1 | +0.0000998654 |
| 2 | +0.0000735911 |
| 3 | +0.0000156148 |
| 4 | -0.0000000033 |
| 5 | -0.0000277019 |

Формально положительны 3/5 folds; четвёртый практически равен контролю, пятый
отрицателен. Paired stratified bootstrap на 300 повторах дал interval
`[-0.0000088; +0.0000691]`; положительны `281/300` повторов (`93.67%`). Нижняя
граница пересекает ноль.

Среди новых base-фич наибольший mean gain получили вторая цифра weekend screen
time (`10014.9`), daily screen time (`6987.1`) и sleep (`2725.6`). Затем идут
social (`1117.8`), gaming (`679.2`) и work/study (`392.9`). Importance показывает
использование, но не устраняет нестабильность OOF.

## Решение

Старый сигнал частично воспроизвёлся: направление снова положительное, base
действительно улучшился. Однако общий выигрыш всего `+0.0000325`, один fold
отрицателен, а bootstrap interval пересекает ноль. По 5-fold критерию пакет
**не принимается** в канонический baseline.

Текущей точкой остаётся `trial-051` с OOF `0.9660227664`. Predictions
`trial-052` сохраняются как weak-positive кандидат: к цифрам можно вернуться при
заранее согласованной проверке веса blend или составе будущего пакета, но нельзя
считать их подтверждённым улучшением.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-052 --candidate lightweight_baseline_plus_second_decimal_digits --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --log-every 50 --bootstrap-repeats 300
```

Артефакты: `artifacts/metrics_trial-052_feature.json`,
`artifacts/oof_predictions_trial-052.csv`,
`artifacts/feature_importance_trial-052.csv` и лог
`20260822T202358418626+0300_cv-feature-trial_trial-052.log`.
