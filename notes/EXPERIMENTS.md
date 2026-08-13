# Experiment registry

| ID | Основное изменение | Local ROC AUC | Kaggle score | Статус |
|---|---|---:|---:|---|
| [`exp-000`](experiments/exp-000-catboost-baseline.md) | Первый CatBoost baseline | **0.955008** | **0.95663** | completed |
| [`exp-001`](experiments/exp-001-cv-gradient-boosting-ensemble.md) | 3-fold CatBoost + LightGBM rank ensemble | **0.963154 OOF** | **0.96466** | completed |

`exp-001` улучшил leaderboard score на `+0.00803`. Следующий итоговый ID —
`exp-002`; сначала должна пройти новая серия согласованных trial.
