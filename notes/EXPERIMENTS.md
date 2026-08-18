# Experiment registry

| ID | Основное изменение | Local ROC AUC | Kaggle score | Статус |
|---|---|---:|---:|---|
| [`exp-000`](experiments/exp-000-catboost-baseline.md) | Первый CatBoost baseline | **0.955008** | **0.95663** | completed |
| [`exp-001`](experiments/exp-001-cv-gradient-boosting-ensemble.md) | 3-fold CatBoost + LightGBM rank ensemble | **0.963154 OOF** | **0.96466** | completed |
| [`exp-002`](experiments/exp-002-max-lightgbm.md) | 5-fold high-budget pure LightGBM с принятыми FE | **0.964047 OOF** | pending | completed locally |
| [`exp-003`](experiments/exp-003-exact-value-lightgbm.md) | Exact-value categories, LR 0.01 и fold early stopping | **0.965044 OOF** | **0.96681** | completed |
| [`exp-004`](experiments/exp-004-high-capacity-exact-value-lightgbm.md) | Конфигурация `exp-002` с exact-value categories | **0.964587 OOF** | **0.96668** | completed; rejected vs `exp-003` |
| [`exp-005`](experiments/exp-005-exp003-exp002-rank-blend.md) | 60/40 rank blend `exp-003` + `exp-002` | **0.966874 OOF** | **0.96827** | completed |
| [`exp-006`](experiments/exp-006-full-exact-catboost.md) | Full 5-fold CatBoost, LR 0.01, 3000 деревьев, без `gaming_hours_exact` | **0.965896 OOF** | **0.96711** | completed; best standalone, rejected vs `exp-005` |

`exp-005` остаётся общим локальным и leaderboard-лидером. `exp-006` — лучший
standalone, но он ниже бленда. Следующий итоговый ID — `exp-007`.
