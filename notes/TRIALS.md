# Trial registry

| ID | Единственное основное изменение | Local ROC AUC | Δ к контролю | Решение |
|---|---|---:|---:|---|
| [`trial-001`](trials/trial-001-catboost-1000-iterations.md) | CatBoost: 400 → 1000 итераций | **0.960705** | **+0.005697** | accepted |
| [`trial-002`](trials/trial-002-catboost-depth-8-screening.md) | При 200 деревьях: depth 7 → 8 | **0.950048** | **+0.001790** | promising; full-budget check required |

Следующий ID: `trial-003`. Он не выбран и не реализуется без предварительного
обсуждения и явного разрешения участника.
