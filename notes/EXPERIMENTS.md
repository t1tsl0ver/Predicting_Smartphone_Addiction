# Experiment registry

| ID | Основное изменение | Local ROC AUC | Kaggle score | Статус |
|---|---|---:|---:|---|
| [`exp-000`](experiments/exp-000-catboost-baseline.md) | Первый CatBoost baseline | **0.955008** | **0.95663** | completed |
| [`exp-001`](experiments/exp-001-cv-gradient-boosting-ensemble.md) | 3-fold CatBoost + LightGBM rank ensemble | **0.963154 OOF** | **0.96466** | completed |
| [`exp-002`](experiments/exp-002-max-lightgbm.md) | 5-fold high-budget pure LightGBM с принятыми FE | **0.964047 OOF** | pending | completed locally |
| [`exp-003`](experiments/exp-003-exact-value-lightgbm.md) | Exact-value categories, LR 0.01 и fold early stopping | **0.965044 OOF** | pending | completed locally |

`exp-003` — новый локальный лидер; его Kaggle score пока не сообщён. Следующий
итоговый ID — `exp-004`.
