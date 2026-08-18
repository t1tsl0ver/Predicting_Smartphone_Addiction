# trial-038 — Notifications value frequency

- Статус: **accepted for screening; positive paired OOF and won 2/2 folds**
- Дата: 2026-08-18
- Контроль: LightGBM exact/base 60/40 blend из `trial-023`

## Изменение

Изолированно добавлена fold-safe частота точного `notifications_per_day`.
Mapping строится только по train fold, unseen validation получает `0`, пропуск —
отдельное значение, target не используется. Остальной протокол неизменен.

## Результаты

| Вариант | Delta к контролю |
|---|---:|
| Exact LightGBM | +0.0000142372 |
| Base LightGBM | **+0.0009377073** |
| 60/40 blend | **+0.0000891784** |

Blend OOF: `0.9645386677 → 0.9646278462`. Fold-deltas:
`+0.0001105224` и `+0.0000680537`.

Paired bootstrap на 100 повторах размера test:

```text
95% interval:      [+0.0000530, +0.0001364]
positive repeats:  100/100
```

Spearman control/candidate blend равен `0.999116`. Exact почти не изменился, но
base получил крупный прирост: frequency представляет дискретную структуру
уведомлений без categorical exact-копии и полезно меняет итоговый blend.

## Вывод

`notifications_per_day_value_frequency` **принята как одиночный screening-
кандидат**: положительны общий OOF, оба folds и весь paired interval. Это пока
не 5-fold подтверждение и не разрешение на комбинацию.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-038 --candidate notifications_per_day_value_frequency
```

Артефакты: `artifacts/metrics_trial-038_feature.json`, OOF, importance и лог
`20260818T125337886747+0300_cv-feature-trial_trial-038.log`.
