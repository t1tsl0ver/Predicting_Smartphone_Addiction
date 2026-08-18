# trial-042 — Missing-pattern category

- Статус: **rejected; large negative paired OOF and lost 2/2 folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Изменение

Полная 12-битная маска пропусков добавлена в оба компонента как одна categorical-
фича с 2925 наблюдаемыми значениями:

```text
missing_pattern_category = categorical(full missing bitmask)
```

Частота из `trial-041` и count из `trial-031` не добавляются. Target не
используется; остальной протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | -0.0000846124 |
| Base LightGBM | **-0.0016285753** |
| 60/40 blend | **-0.0003882542** |

Blend OOF: `0.9645386677 → 0.9641504135`. Fold-deltas:
`-0.0003256671` и `-0.0004506717`.

Paired bootstrap на 100 повторах размера test полностью отрицателен:
`[-0.0004846, -0.0002908]`, положительных повторов `0/100`.

Первый технический запуск остановился до обучения из-за форматирования строки в
диагностическом логе. Logger исправлен и покрыт тестом; приведённые результаты
получены полным успешным перезапуском с неизменной модельной конфигурацией.

## Вывод

`missing_pattern_category` **уверенно отклонена**. Высококардинальная identity
маски особенно сильно повредила base-модели; вероятное объяснение — лёгкое
запоминание редких комбинаций без переносимого target-сигнала. Это завершает
согласованную серию одиночных feature-кандидатов.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-042 --candidate missing_pattern_category
```

Артефакты: `artifacts/metrics_trial-042_feature.json`, OOF, importance и успешный
лог `20260818T130334061271+0300_cv-feature-trial_trial-042.log`.
