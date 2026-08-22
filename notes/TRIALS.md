# Trial registry

| ID | Единственное основное изменение | Local ROC AUC | Δ к контролю | Решение |
|---|---|---:|---:|---|
| [`trial-001`](trials/trial-001-catboost-1000-iterations.md) | CatBoost: 400 → 1000 итераций | **0.960705** | **+0.005697** | accepted |
| [`trial-002`](trials/trial-002-catboost-depth-8-screening.md) | При 200 деревьях: depth 7 → 8 | **0.950048** | **+0.001790** | promising; full-budget check required |
| [`trial-003`](trials/trial-003-catboost400-lightgbm-blend.md) | LightGBM и rank blend с CatBoost d7/400 | **0.962871** (LightGBM) | **+0.007863** к CatBoost 400 | LightGBM accepted; blend rejected |
| [`trial-004`](trials/trial-004-catboost-depth8-1000.md) | При 1000 деревьях: depth 7 → 8 | **0.961378** | **+0.000674** | accepted |
| [`trial-005`](trials/trial-005-catboost1000-lightgbm-blend.md) | Rank blend CatBoost d7/1000 + LightGBM | **0.962857** | **-0.000014** к LightGBM | blend rejected |
| [`trial-006`](trials/trial-006-catboost-depth8-lightgbm-blend.md) | Rank blend CatBoost d8/1000 + LightGBM | **0.962918** | **+0.000047** к LightGBM | weak positive; CV required |
| [`trial-007`](trials/trial-007-three-fold-oof-ensemble.md) | 3-fold OOF: фиксированный blend 15/85 | **0.963154** OOF | **+0.000075** к LightGBM OOF | accepted; won 3/3 folds |
| [`trial-008`](trials/trial-008-pure-lightgbm-500-cv.md) | Новый быстрый baseline: чистый LightGBM/500, 5-fold CV | **0.961091** OOF | **-0.001988** к LGBM/3000 | adopted as fast baseline |
| [`trial-009`](trials/trial-009-drop-gender-stress.md) | Удалены `gender` и `stress_level` | **0.961371** OOF | **+0.000283** | accepted; won 3/5 folds |
| [`trial-010`](trials/trial-010-other-screen-time.md) | Добавлен residual `other_screen_time_hours` | **0.961775** OOF | **+0.000404** | accepted; won 4/5 folds |
| [`trial-011`](trials/trial-011-screen-awake-share.md) | Добавлен `screen_awake_share` | **0.961707** OOF | **-0.000068** | rejected; won 3/5 but total delta negative |
| [`trial-012`](trials/trial-012-screen-social-load.md) | Добавлен product `screen_social_load` | **0.962085** OOF | **+0.000310** | accepted; won 3/5 folds |
| [`trial-013`](trials/trial-013-restore-gender.md) | Изолированно возвращён `gender` | **0.961979** OOF | **-0.000106** | rejected; won 3/5 but total delta negative |
| [`trial-014`](trials/trial-014-high-stress-other-screen-per-sleep.md) | Добавлен High-stress residual/sleep interaction | **0.962243** OOF | **+0.000159** | accepted after LB 0.96412 > 0.96404 |
| [`trial-015`](trials/trial-015-overall-screen-pressure.md) | Добавлен normalized daily/weekend pressure | **0.959576** OOF | **-0.002668** | rejected; lost 5/5 folds |
| [`trial-016`](trials/trial-016-core-screen-pressure.md) | В pressure добавлен отдельный вес social media | **0.961838** OOF | **-0.000405** | rejected; lost 4/5 folds |
| [`trial-017`](trials/trial-017-stress-weekend-ratio.md) | Возврат stress + weekend/daily ratio | **0.961978** OOF | **-0.000265** | rejected; won 4/5 but total delta negative |
| [`trial-018`](trials/trial-018-drop-academic-work-impact.md) | Удалён `academic_work_impact` | **0.962082** OOF | **-0.000161** | rejected; won 3/5 but total delta negative |
| [`trial-019`](trials/trial-019-numeric-exact-categories.md) | Добавлены categorical-копии точных числовых значений | **0.964025** OOF | **+0.001782** | accepted; won 5/5, paired CI above zero |
| [`trial-020`](trials/trial-020-cat-smooth-20.md) | LightGBM `cat_smooth`: 10 → 20 | **0.963977** OOF | **-0.000048** | rejected; won 3/5, paired CI crosses zero |
| [`trial-021`](trials/trial-021-250-trees.md) | Exact-value LightGBM: 500 → 250 деревьев | **0.964279** OOF | **+0.000254** | accepted; won 5/5, paired CI above zero |
| [`trial-022`](trials/trial-022-catboost-lightgbm-200-head-to-head.md) | Standalone CatBoost против LightGBM при одинаковых 200 деревьях | **0.965049** CatBoost OOF | **+0.000756** к LGBM/200 | accepted; CatBoost won 5/5, paired CI above zero |
| [`trial-023`](trials/trial-023-two-fold-family-blends.md) | 2-fold exact/base rank blends внутри CatBoost и LightGBM | **0.964539** LGBM blend OOF | **+0.001713** к LGBM exact; Cat blend -0.001895 | LGBM blend signal accepted; CatBoost 60/40 rejected |
| [`trial-024`](trials/trial-024-drop-gaming-hours-exact.md) | Удалён `gaming_hours_exact`; exact CatBoost против LGBM blend | **0.964589** CatBoost OOF | Cat +0.000073; LGB blend +0.000018 | accepted for CatBoost screening; unproven for LGBM |
| [`trial-025`](trials/trial-025-notifications-per-screen-hour.md) | Добавлен `notifications_per_screen_hour` в оба LGBM-компонента | **0.964550** LGBM blend OOF | +0.000012 | rejected; mixed folds, interval crosses zero |
| [`trial-026`](trials/trial-026-app-opens-per-screen-hour.md) | Добавлен `app_opens_per_screen_hour` в оба LGBM-компонента | **0.964515** LGBM blend OOF | -0.000023 | rejected; negative point estimate, mixed folds |
| [`trial-027`](trials/trial-027-app-opens-per-notification.md) | Добавлен `app_opens_per_notification` в оба LGBM-компонента | **0.964539** LGBM blend OOF | +0.000001 | rejected; effectively zero, mixed folds |
| [`trial-028`](trials/trial-028-entertainment-screen-share.md) | Добавлен `(social + gaming) / daily` в оба LGBM-компонента | **0.964657** LGBM blend OOF | **+0.000118** | accepted for screening; won 2/2, interval positive |
| [`trial-029`](trials/trial-029-work-study-screen-share.md) | Добавлен `work_study / daily` в оба LGBM-компонента | **0.964991** LGBM blend OOF | **+0.000452** | accepted for screening; strong 2/2 and positive interval |
| [`trial-030`](trials/trial-030-weekend-minus-daily-screen.md) | Добавлен `weekend - daily` в оба LGBM-компонента | **0.964505** LGBM blend OOF | -0.000033 | rejected; lost 2/2 folds |
| [`trial-031`](trials/trial-031-missing-feature-count.md) | Добавлено число пропусков исходных фич | **0.964567** LGBM blend OOF | +0.000028 | rejected; mixed folds, interval crosses zero |
| [`trial-032`](trials/trial-032-age-value-frequency.md) | Fold-safe frequency exact-значения `age` | **0.964514** LGBM blend OOF | -0.000025 | rejected; negative, mixed folds |
| [`trial-033`](trials/trial-033-daily-screen-time-value-frequency.md) | Fold-safe frequency exact-значения daily screen time | **0.964586** LGBM blend OOF | **+0.000048** | weak accepted screening; 2/2, interval barely positive |
| [`trial-034`](trials/trial-034-social-media-value-frequency.md) | Fold-safe frequency exact-значения social media hours | **0.964388** LGBM blend OOF | -0.000150 | rejected; lost 2/2, interval negative |
| [`trial-035`](trials/trial-035-gaming-value-frequency.md) | Fold-safe frequency exact-значения gaming hours | **0.964526** LGBM blend OOF | -0.000012 | rejected; negative, mixed folds |
| [`trial-036`](trials/trial-036-work-study-value-frequency.md) | Fold-safe frequency exact-значения work/study hours | **0.964554** LGBM blend OOF | +0.000016 | rejected; mixed folds, interval crosses zero |
| [`trial-037`](trials/trial-037-sleep-value-frequency.md) | Fold-safe frequency exact-значения sleep hours | **0.964498** LGBM blend OOF | -0.000041 | rejected; lost 2/2, interval negative |
| [`trial-038`](trials/trial-038-notifications-value-frequency.md) | Fold-safe frequency exact-значения notifications/day | **0.964628** LGBM blend OOF | **+0.000089** | accepted screening; 2/2 and positive interval |
| [`trial-039`](trials/trial-039-app-opens-value-frequency.md) | Fold-safe frequency exact-значения app opens/day | **0.964568** LGBM blend OOF | +0.000030 | rejected; one fold negative, interval crosses zero |
| [`trial-040`](trials/trial-040-weekend-value-frequency.md) | Fold-safe frequency exact-значения weekend screen time | **0.964567** LGBM blend OOF | +0.000029 | rejected; one fold negative, interval crosses zero |
| [`trial-041`](trials/trial-041-missing-pattern-frequency.md) | Fold-safe частота полной 12-битной маски пропусков | **0.964540** LGBM blend OOF | +0.000002 | rejected; effectively zero, mixed folds |
| [`trial-042`](trials/trial-042-missing-pattern-category.md) | Полная 12-битная missing-mask как categorical-фича | **0.964150** LGBM blend OOF | -0.000388 | rejected; lost 2/2, interval negative |
| [`trial-043`](trials/trial-043-combine-four-screening-winners.md) | Одновременно добавлены четыре победителя одиночной feature-серии | **0.965240** LGBM blend OOF | **+0.000701** | accepted screening; won 2/2, also beats best single |
| [`trial-044`](trials/trial-044-notifications-target-encoding.md) | Leak-safe target encoding точного notifications/day поверх `trial-043` | **0.965142** LGBM blend OOF | **-0.000098** | rejected; lost 2/2 to `trial-043` |
| [`trial-045`](trials/trial-045-all-target-encodings.md) | Leak-safe target encoding всех 15 смысловых фич поверх `trial-043` | **0.965572** LGBM blend OOF | **+0.000332** | accepted screening; won 2/2, interval positive |
| [`trial-046`](trials/trial-046-prune-low-gain-target-encodings.md) | Одним блоком удалены шесть low-gain TE из `trial-045` | **0.965542** LGBM blend OOF | **-0.000029** | rejected; full 15-TE bundle retained |
| [`trial-047`](trials/trial-047-standalone-ebm-kaggle.md) | Standalone EBM против standalone LightGBM, без TE | not run | — | cancelled by participant before Kaggle run |
| [`trial-048`](trials/trial-048-standalone-pyboost-kaggle.md) | Standalone Py-Boost против standalone LightGBM, без TE | not run | — | cancelled by participant before Kaggle run |
| [`trial-049`](trials/trial-049-first-decimal-digits.md) | Первые десятичные цифры шести часовых колонок поверх `trial-043` | 0.965193 LGBM blend OOF | -0.000047 | rejected; lost 2/2, residual check negative |
| [`trial-050`](trials/trial-050-second-decimal-digits.md) | Вторые десятичные цифры шести часовых колонок поверх `trial-043` | **0.965298** LGBM blend OOF | **+0.000058** | accepted screening; 2/2, intervals positive |
| [`trial-051`](trials/trial-051-five-feature-lightweight-5fold.md) | 5-fold: четыре screening-фичи + mean surprisal числовых значений | **0.966023** LGBM blend OOF | **+0.000674** | accepted lightweight baseline; won 5/5, interval positive |
| [`trial-052`](trials/trial-052-second-decimal-digits-5fold.md) | Вторые десятичные цифры поверх лёгкого 5-fold `trial-051` | 0.966055 LGBM blend OOF | +0.000033 | weak positive; not adopted, interval crosses zero |
| [`trial-053`](trials/trial-053-social-within-entertainment.md) | Доля social внутри social + gaming поверх `trial-051` | 0.966038 LGBM blend OOF | +0.000015 | rejected; 2/5, interval crosses zero |
| [`trial-054`](trials/trial-054-social-screen-share.md) | Доля social в daily screen time поверх `trial-051` | **0.966217** LGBM blend OOF | **+0.000195** | accepted; won 5/5, interval positive |

Следующий ID: `trial-055`. По решению участника активная лёгкая линия теперь
использует 5-fold CV при прежних 200 деревьях. LightGBM-контроль — exact/base
rank blend 60/40 с seed `2026`; его актуальная принятая точка — `trial-051` с
OOF `0.9660227664`. Исторические 2-fold trial и `src/trial_config.py` остаются
для воспроизводимости прежней screening-серии.

Новая серия одиночных feature-trial использует тот же cached LightGBM exact/base
60/40 контроль `trial-023`. Каждый кандидат проверяется отдельно; комбинации не
запускались до отдельного решения участника. После полного рейтинга `trial-043`
объединил ровно четыре прошедших screening кандидата и получил `+0.000701` к
контролю. `trial-044` добавил leak-safe target encoding notifications/day, но
проиграл пакету на `-0.000098` и был отклонён. `trial-045` добавил согласованный
набор TE-копий всех 15 смысловых фич и выиграл `+0.000332`, в основном за счёт
резкого усиления base-компонента. Единственная групповая чистка `trial-046`
ухудшила blend на `-0.000029`. После `exp-008` TE-направление закрыто решением
участника: следующие модели сохраняют четыре обычные фичи `trial-043`, но не
используют 15 TE-копий. Подготовленные Kaggle-проверки `trial-047/048` отменены
участником до запуска и результатов не имеют. Новые model-family направления
закрыты; дальнейшие кандидаты снова проверяются как изолированные LightGBM
feature-trial после отдельного обсуждения. `trial-049` отклонил первые
десятичные цифры. Независимый `trial-050` с вторыми цифрами дал `+0.000058` и
выиграл 2/2 folds, но затем участник выбрал другой групповой 5-fold кандидат.
`trial-051` одновременно подтвердил четыре обычные screening-фичи и добавил
mean surprisal точных числовых значений: пакет дал `+0.000674`, выиграл 5/5
folds и стал текущим лёгким baseline. Отдельный вклад surprisal не установлен.
`trial-052` добавил вторые десятичные цифры уже поверх этого baseline. Base вырос
на `+0.000536`, но blend — только на `+0.000033`; один fold отрицателен, interval
пересекает ноль, поэтому каноническая точка не изменена.
