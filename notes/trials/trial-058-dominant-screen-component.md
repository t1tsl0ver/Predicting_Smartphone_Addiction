# trial-058 — Доминирующий компонент экранного времени

- Статус: **rejected**
- Дата: 2026-08-22
- Контроль: `trial-051`, OOF `0.9660227661`

## Вопрос и гипотеза

Может ли категориальная метка крупнейшей доли `social`, `gaming`, `work_study`
или `other` выделить режимы генератора? Метка строится только по значениям
строки; невозможные или неполные разложения остаются missing.

## Изменение и контроль

Добавлена только categorical-фича `dominant_screen_component`; конфигурация —
5 folds, seed `2026`, LightGBM 200 деревьев, LR `0.10`, blend `60/40`.

## Результаты

Exact delta `-0.0001906841`, base `-0.0002400399`, blend
**`-0.0001817042`** до `0.9658410619`. Fold-delta: `-0.0001326090`,
`-0.0003116702`, `-0.0000888361`, `-0.0002920208`, `-0.0000858825` —
проигрыш 5/5. Bootstrap CI95 `[-0.0002530400; -0.0000918551]`,
`P(delta>0)=0.00`, 100 повторов. Время: `214.5 s`.

## Вывод

Грубая доминирующая категория устойчиво уничтожает информацию. Фича отклонена.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-058 --candidate dominant_screen_component --include-lightweight-baseline --control-oof artifacts\oof_predictions_trial-051.csv --control-metrics artifacts\metrics_trial-051_feature.json --folds 5 --iterations 200 --bootstrap-repeats 100 --log-every 100
```

