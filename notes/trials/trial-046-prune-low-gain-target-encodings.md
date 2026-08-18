# trial-046 — Групповая чистка low-gain target encodings

- Статус: **rejected; full trial-045 bundle retained**
- Дата: 2026-08-18
- Контроль: полный пакет 15 target encodings `trial-045`

## Гипотеза и заранее заданное решение

Шесть TE-копий получили очень малый gain в `trial-045`. Их совместное удаление
может упростить модель без потери переносимого сигнала. По решению участника это
единственная попытка чистки перед полным экспериментом:

- если OOF остаётся тем же или улучшается — берём сокращённый набор;
- если OOF снижается — возвращаем полный `trial-045`;
- дополнительных разбиений группы и перебора не проводим.

## Единственное изменение

Из полного пакета удалены:

```text
academic_work_impact_target_encoding
work_study_screen_share_target_encoding
entertainment_screen_share_target_encoding
other_screen_time_hours_target_encoding
screen_social_load_target_encoding
high_stress_other_screen_per_sleep_target_encoding
```

Остались девять TE-копий исходных числовых признаков. Четыре screening-фичи,
модель, 60/40 blend, inner/outer folds, smoothing `alpha=50` и seed не менялись.

## Результаты против trial-045

| Вариант | Control OOF | Pruned OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.963739498 | 0.963629232 | **-0.000110266** |
| Base LightGBM | 0.965155402 | 0.965162246 | +0.000006845 |
| 60/40 blend | **0.965571690** | 0.965542263 | **-0.000029427** |

Blend проиграл оба folds: `-0.000006089` и `-0.000052655`. Paired bootstrap на
100 повторах пересёк ноль: `[-0.000077481; +0.000029301]`, положительны
`19/100` повторов. Ухудшение небольшое и статистически неустойчивое, но оно
отрицательно по point estimate и на обоих folds.

## Вывод

По заранее заданному правилу чистка **отклонена**. Для полного эксперимента
используется исходный пакет `trial-045` со всеми 15 TE-копиями. Дополнительный
поиск состава не проводится, чтобы не расходовать лимит и не подгоняться под
быстрые два folds.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-046 --candidate screening_winners_plus_pruned_target_encodings --control-oof artifacts\oof_predictions_trial-045.csv --control-metrics artifacts\metrics_trial-045_feature.json --target-encoding-alpha 50 --target-encoding-inner-folds 5
```

Артефакты: `artifacts/metrics_trial-046_feature.json`, OOF, importance и лог
`20260818T141114616632+0300_cv-feature-trial_trial-046.log`.
