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
| [`exp-007`](experiments/exp-007-full-target-encoded-lightgbm.md) | Full 5-fold exact/base LightGBM с 15 leak-safe TE | **0.966831 OOF** | not submitted | completed locally; retained for blend |
| [`exp-008`](experiments/exp-008-exp005-exp007-rank-blend.md) | 50/50 rank blend `exp-005` + `exp-007` | **0.967339 OOF** | **0.96863** | completed; current leader |
| [`exp-009`](experiments/exp-009-full-trial066-no-te-lightgbm.md) | Full 5-fold exact/base LightGBM с семью no-TE фичами `trial-066` | **0.967137 OOF** | not submitted | completed locally; best standalone no-TE |
| [`exp-010`](experiments/exp-010-exp009-exp007-rank-blend.md) | 55/45 rank blend `exp-009` + `exp-007` | **0.967528 OOF** | **0.96874** | completed; current leader |
| [`exp-011`](experiments/exp-011-extended-base-budget-lightgbm.md) | Cached exact `exp-009` + base LightGBM 8000/800, fixed 60/40 | **0.967541 OOF** | not submitted | completed locally; best autonomous no-TE |

`exp-011` — текущий локальный standalone-лидер; действующий leaderboard-лидер —
`exp-010` с Kaggle `0.96874`. Увеличенный base-бюджет дал `+0.000404` к
`exp-009` и выиграл 5/5. Следующий итоговый ID — `exp-012`.
