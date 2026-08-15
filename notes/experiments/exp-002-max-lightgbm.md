# exp-002 — High-budget pure LightGBM

- Статус: **completed locally; Kaggle score pending**
- Дата: 2026-08-14
- Контроль: `trial-014`
- Модель: только LightGBM, без ансамбля

## Цель

Обучить максимально сильную в разумном вычислительном бюджете версию чистого
LightGBM на всех принятых feature-engineering улучшениях и создать Kaggle
submission как отдельный итоговый эксперимент, а не очередной trial.

## Принятые фичи

- исходные признаки без `gender` и `stress_level`;
- `other_screen_time_hours` из `trial-010`;
- `screen_social_load` из `trial-012`;
- `high_stress_other_screen_per_sleep` из `trial-014`.

Отклонённые `screen_awake_share`, `overall_screen_pressure` и
`core_screen_pressure` не используются.

## Конфигурация

```text
folds=5
shuffle=True
seed=2026
n_estimators=8000 maximum per fold
learning_rate=0.02
num_leaves=127
max_depth=-1
min_child_samples=100
reg_alpha=0.0
reg_lambda=5.0
early_stopping_rounds=400
log_every=100
```

В каждом fold сохраняется лучшая по validation AUC итерация. Test predictions
пяти fold-моделей усредняются отдельно как raw probabilities и percentile ranks;
для submission выбирается вариант с более высоким OOF AUC.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_lightgbm.py --run-id exp-002 --iterations 8000 --learning-rate 0.02 --num-leaves 127 --max-depth -1 --min-child-samples 100 --reg-lambda 5 --early-stopping-rounds 400 --log-every 100 --drop-features gender stress_level --add-other-screen-time --add-screen-social-load --add-high-stress-other-screen-per-sleep --submission submission.csv
```

## Результат

| Метрика | `trial-014` | `exp-002` | Delta |
|---|---:|---:|---:|
| Raw OOF ROC AUC | 0.9622430857 | **0.9640427056** | **+0.0017996200** |
| Fold-rank OOF ROC AUC | 0.9622452251 | **0.9640472606** | **+0.0018020354** |

Для submission выбраны fold ranks.

### Fold-wise сравнение

| Fold | `trial-014` | `exp-002` | Delta | Best iteration |
|---:|---:|---:|---:|---:|
| 1 | 0.9618137470 | **0.9633614731** | **+0.0015477260** | 2033 |
| 2 | 0.9628245113 | **0.9646593982** | **+0.0018348869** | 2527 |
| 3 | 0.9632180879 | **0.9649076832** | **+0.0016895953** | 2383 |
| 4 | 0.9622383879 | **0.9640764927** | **+0.0018381048** | 3229 |
| 5 | 0.9611308178 | **0.9632312474** | **+0.0021004295** | 2723 |

Усиленная конфигурация выиграла на всех 5 folds. Средняя лучшая итерация —
`2579`, диапазон `2033–3229`; потолок 8000 оказался достаточным с большим
запасом, а early stopping предотвратил лишнее обучение.

Фактическое время CV: `19:47`, полное время запуска с подготовкой и сохранением:
`19:52`. По живому логу ETA после первого fold был около 14 минут, затем вырос до
19–20 минут из-за длительных predictions моделей с 2000–3200 деревьями.

## Submission

```text
file=submission.csv
rows=296302
prediction=mean fold percentile ranks
min=0.000003374935
max=0.999954775871
sha256=dceb10f2e580b545b860589121608bd4b55194df32a38f745a64fe82db170395
Kaggle score=pending
```

## Вывод

Большой бюджет, меньший learning rate, 127 leaves и регуляризация дали сильный и
стабильный прирост `+0.0018020` OOF относительно принятого быстрого контроля.
Это новый локальный лидер проекта. Окончательный статус leaderboard будет
записан после загрузки submission на Kaggle.
