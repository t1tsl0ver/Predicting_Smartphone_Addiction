# trial-054 — Доля social в общем экранном времени

- Статус: **accepted; current lightweight baseline**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Сигнализирует ли генератор не только абсолютными social hours, но и их долей в
общем экранном бюджете? Проверяем `social_media_hours /
daily_screen_time_hours`; target и статистики датасета не используются.

## Изменение и контроль

К пяти фичам `trial-051` добавлена только `social_screen_share`. Нулевой
знаменатель и пропуски дают `NaN`. Неизменны 5 stratified folds, seed `2026`,
LightGBM 200 деревьев, LR `0.10` и exact/base fold-rank blend `60/40`.

## Результаты

| Компонент | OOF delta |
|---|---:|
| Exact | **+0.0003305239** |
| Base | **+0.0000841025** |
| Blend | **+0.0001946045** до **`0.9662173706`** |

Fold-delta blend: `+0.0002235476`, `+0.0002277398`, `+0.0000590224`,
`+0.0001770365`, `+0.0002859972`: победа 5/5. Paired bootstrap 100 повторов:
CI95 `[+0.0000958680; +0.0002808130]`, `P(delta>0)=1.00`. Время: `200.1 s`.

## Наблюдения и вывод

Плюс есть в обеих компонентах, на каждом fold и во всём bootstrap interval.
Фича принята. Это единственный строгий победитель серии `trial-053..061`, поэтому
«объединение всех положительных» совпадает с этим trial и повторно не обучается.
Новый лёгкий baseline отстаёт от no-TE `exp-005` на `0.0006569974` OOF.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-054 --candidate social_screen_share --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

