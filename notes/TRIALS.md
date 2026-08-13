# Trial registry

| ID | Единственное основное изменение | Local ROC AUC | Δ к контролю | Решение |
|---|---|---:|---:|---|
| [`trial-001`](trials/trial-001-catboost-1000-iterations.md) | CatBoost: 400 → 1000 итераций | **0.960705** | **+0.005697** | accepted |
| [`trial-002`](trials/trial-002-catboost-depth-8-screening.md) | При 200 деревьях: depth 7 → 8 | **0.950048** | **+0.001790** | promising; full-budget check required |
| [`trial-003`](trials/trial-003-catboost400-lightgbm-blend.md) | LightGBM и rank blend с CatBoost d7/400 | **0.962871** (LightGBM) | **+0.007863** к CatBoost 400 | LightGBM accepted; blend rejected |
| [`trial-004`](trials/trial-004-catboost-depth8-1000.md) | При 1000 деревьях: depth 7 → 8 | **0.961378** | **+0.000674** | accepted |
| [`trial-005`](trials/trial-005-catboost1000-lightgbm-blend.md) | Rank blend CatBoost d7/1000 + LightGBM | **0.962857** | **-0.000014** к LightGBM | blend rejected |

Следующий ID: `trial-006`. Дешёвая проверка CatBoost d8/1000 + LightGBM нужна,
поскольку d8 сильнее d7, а его validation predictions уже сохранены.
