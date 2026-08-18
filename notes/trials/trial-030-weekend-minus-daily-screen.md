# trial-030 — Weekend minus daily screen time

- Статус: **rejected; negative OOF and lost 2/2 folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Вопрос и изменение

Помогает ли явное изменение поведения в выходные относительно обычного дня?
Изолированно добавлена одна числовая фича без exact-копии:

```text
weekend_minus_daily_screen =
    weekend_screen_time - daily_screen_time_hours
```

Остальные признаки, два folds, seed `2026`, 200 деревьев, learning rate `0.10`,
exact/base blend `60/40` и cached контроль неизменны. В отличие от `trial-017`,
здесь одновременно не возвращается `stress_level` и не проверяется ratio.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0000802891 |
| Base LightGBM | -0.0000783893 |
| 60/40 blend | **-0.0000331908** |

Blend OOF снизился с `0.9645386677` до `0.9645054769`. Fold-deltas:
`-0.0000022119` и `-0.0000638303`.

Paired bootstrap на 100 повторах размера test дал interval
`[-0.0000923, +0.0000507]`, положительными были 22/100 повторов. Spearman
control/candidate blend равен `0.997198`. Mean gain фичи составил `1354.82` в
exact и `6711.74` в base, но оба компонента ухудшили OOF.

## Вывод

`weekend_minus_daily_screen` **отклонена как одиночный кандидат**: отрицательны
общий OOF, оба folds и оба компонента. Простая разность не раскрыла полезного
остаточного сигнала поверх уже существующих daily/weekend признаков.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-030 --candidate weekend_minus_daily_screen
```

Артефакты: `artifacts/metrics_trial-030_feature.json`, OOF, importance и лог
`20260818T123450340099+0300_cv-feature-trial_trial-030.log`.
