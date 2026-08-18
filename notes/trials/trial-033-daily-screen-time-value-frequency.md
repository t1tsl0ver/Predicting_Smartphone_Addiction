# trial-033 — Daily screen-time value frequency

- Статус: **accepted for screening; weak positive 2/2 and interval above zero**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Изменение

Одна fold-safe числовая фича добавлена в оба компонента:

```text
daily_screen_time_hours_value_frequency =
    count(exact daily_screen_time_hours) / fold train rows
```

Mapping строится только по train-части fold; unseen validation получает `0`,
пропуск является отдельным значением, target не используется. Остальной
протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0000785809 |
| Base LightGBM | **+0.0002473857** |
| 60/40 blend | **+0.0000476784** |

Blend OOF: `0.9645386677 → 0.9645863461`. Fold-deltas:
`+0.0000930057` и `+0.0000023983`.

Paired bootstrap на 100 повторах размера test:

```text
95% interval:      [+0.0000016, +0.0000966]
positive repeats:  97/100
```

Spearman control/candidate blend равен `0.998434`. Frequency почти не
использовалась exact-моделью (mean gain `29.11`), но стала очень заметной для
base (`283802.69`) и улучшила его OOF.

## Вывод

Фича **принята как слабый screening-кандидат** по заранее заданному правилу:
общая дельта положительна, оба folds неотрицательны и paired lower bound выше
нуля. Сигнал пограничный: второй fold почти равен контролю, а нижняя граница
интервала близка к нулю. В рейтинге он должен находиться ниже устойчивых
`trial-028` и `trial-029`; 5-fold подтверждения пока нет.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-033 --candidate daily_screen_time_hours_value_frequency
```

Артефакты: `artifacts/metrics_trial-033_feature.json`, OOF, importance и лог
`20260818T124249785689+0300_cv-feature-trial_trial-033.log`.
