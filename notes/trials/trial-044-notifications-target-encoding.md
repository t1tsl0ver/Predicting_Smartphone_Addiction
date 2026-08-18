# trial-044 — Target encoding для notifications/day

- Статус: **rejected; blend lost 2/2 folds to trial-043**
- Дата: 2026-08-18
- Контроль: пакет четырёх screening-победителей `trial-043`

## Гипотеза

Fold-safe frequency точного `notifications_per_day` уже дала устойчивый прирост
в `trial-038`. Средний target для каждого повторяющегося значения может передать
более прямой supervised-сигнал и дополнить frequency без утечки.

## Единственное изменение

Поверх неизменного пакета `trial-043` добавлена одна числовая фича:

```text
notifications_per_day_target_encoding
```

Encoding защищён двумя уровнями CV:

- для внешней validation mapping строится только по внешнему train-fold;
- обучающие значения модели создаются через 5-fold OOF внутри внешнего train;
- сглаживание зафиксировано заранее: `alpha = 50`;
- missing считается отдельным значением;
- unseen получает global target mean той части, на которой строился mapping.

Собственный target строки не участвует в её encoding. Источник, четыре фичи
контроля, 60/40 вес, outer folds, seed и параметры моделей не менялись.

## Результаты против trial-043

| Вариант | Control OOF | Candidate OOF | Delta |
|---|---:|---:|---:|
| Exact LightGBM | 0.963720876 | 0.963645119 | **-0.000075757** |
| Base LightGBM | 0.960808070 | 0.961188469 | **+0.000380399** |
| 60/40 blend | **0.965239802** | 0.965142180 | **-0.000097622** |

Итоговый blend проиграл на обоих folds: `-0.000069882` и `-0.000125211`.
Основной 100-repeat paired bootstrap дал полностью отрицательный interval
`[-0.000196306; -0.000001870]`, положительны только `2/100` повторов.

Дополнительный 300-repeat bootstrap получил
`[-0.000192129; +0.000005457]`: верхняя граница едва пересекла ноль, положительны
`10/300` повторов. Это не меняет решение: point estimate отрицателен и кандидат
проиграл 2/2 folds.

## Диагностика компонентов

Target encoding получил заметный gain importance (`290232` в base и `87244` в
exact), но это не превратилось в улучшение blend. Base вырос, exact ухудшился, а
их ранги и взаимная корреляция изменились так, что AUC смеси снизился.

Post-hoc диагностика без переобучения — старый exact `trial-043` плюс новый
TE-base — дала `-0.000008758` к контролю, с fold-deltas `+0.000029995` и
`-0.000047232`. Это не отдельный trial и не принятое изменение; результат лишь
показывает, что перенос TE только в base пока тоже не имеет устойчивого сигнала.

Первый полный запуск использовал старый `trial-023` как встроенный контроль и
потому не мог корректно выставить incremental-решение. Runner был расширен для
cached feature-trial controls, после чего тот же детерминированный кандидат
повторён относительно `trial-043`. Итоговые артефакты относятся ко второму
запуску; предсказания и OOF воспроизвелись.

## Вывод

`notifications_per_day_target_encoding` в этой leak-safe конфигурации
**отклонён**. Он не добавляется к пакету `trial-043`. Это не доказывает
бесполезность любого target encoding: другие источники или заранее выбранное
сглаживание должны проверяться отдельными trial, без post-hoc подбора на этом
результате.

## Команды

```powershell
.\.venv\Scripts\python.exe -u src\cv_feature_trial.py --run-id trial-044 --candidate screening_winners_plus_notifications_target_encoding --control-oof artifacts\oof_predictions_trial-043.csv --control-metrics artifacts\metrics_trial-043_feature.json --target-encoding-alpha 50 --target-encoding-inner-folds 5
.\.venv\Scripts\python.exe -u src\compare_oof.py --control artifacts\oof_predictions_trial-043.csv --candidate artifacts\oof_predictions_trial-044.csv --prediction-column candidate_blend_fold_rank --repeats 300 --sample-size 296302 --seed 2026 --output artifacts\comparison_trial-044_vs_trial-043.json
```

Артефакты: `artifacts/metrics_trial-044_feature.json`, OOF, importance,
`artifacts/comparison_trial-044_vs_trial-043.json` и итоговый лог
`20260818T134755749075+0300_cv-feature-trial_trial-044.log`.
