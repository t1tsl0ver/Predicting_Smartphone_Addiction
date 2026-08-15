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

Следующий ID: `trial-020`. Активная линия разработки использует чистый LightGBM
на 500 деревьев с 5-fold CV; CatBoost и ансамбль остаются только историческими
результатами `exp-001`.
