# trial-056 — Энтропия состава экранного времени

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Может ли генератор различать равномерный и концентрированный экранный бюджет?
Фича — entropy долей social, gaming, work/study и остатка `other` в daily screen
time. Неполные или физически отрицательные разложения дают `NaN`.

## Изменение и контроль

Добавлена только `screen_mix_entropy`; 5 folds, seed `2026`, LightGBM 200
деревьев, LR `0.10` и blend `60/40` совпадают с `trial-051`.

## Результаты

Exact delta `-0.0000377769`, base `+0.0000059454`, blend
**`-0.0000174283`** до `0.9660053378`. Fold-delta: `+0.0000320277`,
`-0.0000788611`, `+0.0000463387`, `-0.0000625565`, `-0.0000258090` — 2/5.
Bootstrap CI95 `[-0.0000561413; +0.0000248711]`, `P(delta>0)=0.23`, 100
повторов. Время: `196.0 s`.

## Вывод

Точечная оценка отрицательна, folds смешанные. Энтропия отклонена.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-056 --candidate screen_mix_entropy --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

