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

Следующий ID: `trial-036`. Канонические параметры быстрых trial хранятся в
`src/trial_config.py` и включают 2-fold CV; кандидат перед финальной версией
подтверждается на 5 folds. LightGBM-контроль использует 200 деревьев и exact/base
rank blend 60/40 с полным exact-набором. CatBoost-контроль — самостоятельный
CatBoost/200 без `gaming_hours_exact`; это удаление пока принято только для
screening-линии и требует 5-fold подтверждения перед exp.

Новая серия одиночных feature-trial использует тот же cached LightGBM exact/base
60/40 контроль `trial-023`. Каждый кандидат проверяется отдельно; комбинации не
запускаются до отдельного решения участника. После серии составляется рейтинг
одиночных кандидатов.
