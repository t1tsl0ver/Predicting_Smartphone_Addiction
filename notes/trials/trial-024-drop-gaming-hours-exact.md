# trial-024 — Удаление `gaming_hours_exact`

- Статус: **accepted for CatBoost screening line; not accepted for LightGBM**
- Дата: 2026-08-17
- Контроль: OOF `trial-023` на тех же двух folds
- Режим: быстрый 2-fold trial; перед финальным exp требуется 5-fold confirmation

## Вопрос

Можно ли удалить наименее полезную общую exact-value категорию, одновременно
немного ускорив CatBoost и не ухудшив качество одиночного exact CatBoost и
LightGBM exact/base blend?

## Выбор единственной удаляемой фичи

По 5-fold importance `trial-022` среди девяти `_exact`-копий:

| Фича | CatBoost importance | Доля Cat exact importance | Cat rank снизу | LGBM gain | Доля LGB exact gain | LGB rank снизу |
|---|---:|---:|---:|---:|---:|---:|
| `gaming_hours_exact` | 0.345047 | 0.56% | 1/9 | 31 786.54 | 3.99% | 2/9 |
| `age_exact` | 1.271727 | 2.07% | 3/9 | 6 201.84 | 0.78% | 1/9 |
| `work_study_hours_exact` | 0.676432 | 1.10% | 2/9 | 52 872.24 | 6.64% | 4/9 |

`gaming_hours_exact` имеет лучший совместный низкий rank двух семейств: оно
последнее у CatBoost и второе с конца у LightGBM. Поэтому удаляется только оно.
Importance используется для выбора кандидата, но не считается доказательством
бесполезности; решение принимается по парной OOF-ablation.

## Гипотеза

Сырый `gaming_hours` и производные screen-time признаки уже содержат большую
часть сигнала. Exact-копия с 401 значением может добавлять шум и отдельную
стоимость категориальных статистик. Её удаление способно немного улучшить
обобщение CatBoost и ускорить обучение.

## Дизайн

Канонический trial-конфиг вынесен в `src/trial_config.py`:

```text
folds=2, StratifiedKFold(shuffle=True), seed=2026
iterations=200, early_stopping=False
CatBoost: learning_rate=0.08, depth=8, l2_leaf_reg=5
LightGBM: learning_rate=0.10, num_leaves=31, min_child_samples=20,
          cat_smooth=10, reg_alpha=0, reg_lambda=0
LightGBM blend: 60% candidate exact fold rank + 40% cached base fold rank
```

Скрипт проверяет совпадение train rows, IDs, targets, folds, seed, бюджета
деревьев и веса blend с cached control `trial-023`. Переобучаются только два
candidate CatBoost и два candidate exact-LightGBM. Base-LightGBM predictions
берутся без изменения из `trial-023`.

Основной заранее заданный point-критерий общей ablation — положительная общая
дельта обоих семейств. Дополнительно фиксируются допуск неухудшения `0.00010`,
дельты каждого fold и paired bootstrap. Для переноса в следующую family-specific
trial-линию используется более консервативное требование: общий плюс, победа
2/2 folds и bootstrap interval выше нуля.

## Результат

| Семейство | Control OOF | Candidate OOF | Delta |
|---|---:|---:|---:|
| Exact CatBoost | 0.9645158783 | **0.9645889036** | **+0.0000730253** |
| LightGBM 60/40 blend | 0.9645386677 | **0.9645568287** | **+0.0000181610** |

Формальный общий point-критерий выполнен, но сила подтверждения у семейств
разная.

### Fold-wise

| Fold | Cat control | Cat candidate | Delta | LGB blend control | LGB blend candidate | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9643967987 | **0.9644972266** | **+0.0001004278** | 0.9646319888 | **0.9646877280** | **+0.0000557392** |
| 2 | 0.9646349590 | **0.9646805815** | **+0.0000456225** | **0.9644457185** | 0.9644263071 | **-0.0000194114** |

CatBoost выиграл 2/2 folds. LightGBM blend выиграл первый и немного проиграл
второй.

### Paired bootstrap

100 стратифицированных повторов размера `296302`; интервал отражает только
row-sampling uncertainty и не покрывает повторное обучение, выбор фичи или
смену folds.

| Сравнение | Mean delta | 95% interval | Positive repeats |
|---|---:|---:|---:|
| Cat candidate − control | +0.0000717 | **[+0.0000207, +0.0001177]** | 100/100 |
| LGB blend candidate − control | +0.0000226 | [-0.0000411, +0.0000940] | 65/100 |
| Candidate Cat − candidate LGB blend | +0.0000281 | [-0.0001469, +0.0002449] | 57/100 |

Точечный CatBoost выше candidate LightGBM blend на `+0.0000321`, но folds имеют
разные направления и интервал широко пересекает ноль. Семейства считаются
неразличимыми на этом trial.

## Время

| Модель | Control, 2 folds | Candidate, 2 folds | Изменение |
|---|---:|---:|---:|
| CatBoost exact | 449.31 s | 413.35 s | **-8.0%** |
| LightGBM exact | 20.23 s | 20.92 s | +3.4% |

CatBoost ускорился на обоих folds примерно на 7–9%. LightGBM-разницу порядка
секунды считаем шумом, а не замедлением от удаления фичи. Весь запуск, включая
три bootstrap и сохранение OOF, занял `8:12`; model/CV часть — `7:15`.

## Решение

- Для CatBoost trial-линии удаление `gaming_hours_exact` принято: общий AUC
  вырос, направление повторилось 2/2 folds, row-bootstrap interval выше нуля,
  а обучение стало примерно на 8% быстрее.
- Для LightGBM удаление не принято как улучшение: общая дельта мала, один fold
  отрицательный, bootstrap interval пересекает ноль. В LightGBM exact-компоненте
  фича пока сохраняется.
- Единого общего feature set после trial не навязываем: CatBoost использует 21
  признак без `gaming_hours_exact`, LightGBM blend сохраняет контрольные 22/13.
- Результат CatBoost остаётся screening-сигналом. Перед включением в финальный
  exp ablation должна быть подтверждена на 5 folds; feature-selection uncertainty
  текущий bootstrap не учитывает.
- Submission не создавался.

## Команда воспроизведения

Команда намеренно не содержит `--folds 2`: это часть канонического trial-конфига.

```powershell
.\.venv\Scripts\python.exe -u src\cv_drop_exact_feature.py
```

Артефакты:

- `artifacts/metrics_trial-024_drop_exact.json`;
- `artifacts/oof_predictions_trial-024.csv`;
- `artifacts/feature_importance_trial-024.csv`;
- `artifacts/logs/20260817T175310678034+0300_cv-drop-exact-feature_trial-024.log`.
