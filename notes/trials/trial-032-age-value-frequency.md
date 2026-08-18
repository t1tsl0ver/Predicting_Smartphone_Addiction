# trial-032 — Age value frequency

- Статус: **rejected; negative point estimate and mixed folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Вопрос и гипотеза

Даёт ли частота точного значения возраста сигнал плотности синтетического
генератора, отличный от самого числового и categorical exact-значения?

## Единственное основное изменение

В оба LightGBM-компонента добавлена одна числовая фича:

```text
age_value_frequency = count(age exact value) / fold train rows
```

Encoding выполняется fold-safe: mapping строится только по train-части fold,
validation-значения применяют этот mapping, неизвестные получают `0`. Пропуск
считается отдельным значением. Target не используется. Остальная 2-fold
конфигурация и cached контроль неизменны.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | +0.0000177987 |
| Base LightGBM | -0.0003984694 |
| 60/40 blend | **-0.0000250093** |

Blend OOF снизился с `0.9645386677` до `0.9645136584`. Fold-deltas:
`+0.0000195299` и `-0.0000696955`.

Paired bootstrap на 100 повторах размера test:

```text
95% interval:      [-0.0000663, +0.0000177]
positive repeats:  10/100
```

Spearman control/candidate blend равен `0.998806`. Mean gain frequency-фичи
составил `26.45` в exact и `66580.96` в base-компоненте. Очень высокий base gain
сопровождался сильным ухудшением base OOF.

## Вывод

`age_value_frequency` **отклонена как одиночный кандидат**. Точечный blend
отрицателен, folds расходятся по знаку, paired interval пересекает ноль. Частота
возраста особенно хорошо показывает, что использование фичи деревьями не
доказывает улучшение обобщения.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-032 --candidate age_value_frequency
```

Артефакты: `artifacts/metrics_trial-032_feature.json`, OOF, importance и лог
`20260818T124002807834+0300_cv-feature-trial_trial-032.log`.
