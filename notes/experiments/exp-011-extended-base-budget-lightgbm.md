# exp-011 — Extended base-budget no-TE LightGBM blend

- Статус: **completed locally; best autonomous no-TE model**
- Дата: 2026-08-29
- Контроль: `exp-009`
- Единственное изменение: base budget 3000/400 → 8000/800

## Цель

Проверить наблюдение `exp-009`: exact-компонент стабильно находил плато около
1800–2000 деревьев, тогда как base на всех пяти folds упирался в максимум 3000
при продолжающемся росте validation AUC. Exact OOF/test predictions переиспользованы
побитово; переобучался только base.

## Конфигурация

```text
folds=5, shuffle=True, seed=2026
features=accepted seven-feature trial-066 no-TE bundle
exact=cached from exp-009
base max iterations=8000
base learning_rate=0.01
base early_stopping_rounds=800
exact/base rank blend=0.60/0.40 fixed
```

Features, fold assignments, frequency mappings, model seed и вес blend не
изменялись. Вес 60/40 не переоптимизировался, чтобы измерить только эффект
base-бюджета.

## Команда

```powershell
.\.venv\Scripts\python.exe -u src\cv_target_encoded_blend.py --run-id exp-011 --feature-bundle trial-066-no-te --folds 5 --iterations 3000 --learning-rate 0.01 --early-stopping-rounds 400 --base-iterations 8000 --base-early-stopping-rounds 800 --cached-exact-oof artifacts\oof_predictions_exp-009.csv --cached-exact-test artifacts\test_predictions_exp-009.csv --exact-weight 0.60 --log-every 100
```

## Результат

| Компонент | `exp-009` | `exp-011` | Delta |
|---|---:|---:|---:|
| Exact fold-rank | 0.9667012675 | 0.9667012675 | 0.0000000000 |
| Base fold-rank | 0.9637438759 | **0.9651322744** | **+0.0013883984** |
| Fixed 60/40 blend | 0.9671371213 | **0.9675414515** | **+0.0004043302** |

| Fold | Base delta | Blend delta |
|---:|---:|---:|
| 1 | +0.0013539472 | +0.0003971425 |
| 2 | +0.0012353679 | +0.0003732531 |
| 3 | +0.0014194107 | +0.0004119971 |
| 4 | +0.0015154161 | +0.0004331482 |
| 5 | +0.0014178505 | +0.0004065969 |

Прирост получен на 5/5 folds. Paired stratified bootstrap fixed blend против
`exp-009` на 300 выборках размера test дал mean delta `+0.0004036982`, CI95
`[+0.0003750798, +0.0004328153]` и `300/300` положительных повторов.

`exp-011` самостоятельно оказался на `0.0000132704` выше OOF ансамбля
`exp-010`, но отдельный submission не создавался: следующий заранее
согласованный шаг проверяет его post-hoc blend с `exp-007`.

## Диагностика обучения

| Fold | Best base iteration |
|---:|---:|
| 1 | 7998 |
| 2 | 7996 |
| 3 | 7996 |
| 4 | 7987 |
| 5 | 8000 |

Гипотеза о преждевременной остановке base подтверждена. Все folds снова
выбрали границу нового бюджета, но дополнительная отдача после примерно 7000
уже мала. Поэтому 8000 считается практическим рабочим бюджетом; дальнейшее
увеличение нельзя считать автоматически оправданным.

## Техническая реализация

`src/cv_target_encoded_blend.py` теперь поддерживает независимые iterations,
learning rate и early-stopping patience для exact/base. Опциональный cached-exact
режим строго проверяет row/fold/target/test-ID alignment, finite values и
переиспользует exact OOF/test predictions без повторного обучения.

Проверены `691369` уникальных OOF-строк, `296302` test-строки и побитовое
совпадение cached exact ranks с `exp-009`. Полный base-only CV занял около
29 минут.

## Артефакты

```text
artifacts/metrics_exp-011_full_feature_blend.json
artifacts/oof_predictions_exp-011.csv
artifacts/test_predictions_exp-011.csv
artifacts/feature_importance_exp-011.csv
artifacts/comparison_exp-011_vs_exp-009.json
```

## Вывод

Раздельные бюджеты exact/base необходимы: общий лимит 3000 был подходящим для
exact, но систематически недообучал base. `exp-011` принят как новый сильнейший
автономный no-TE кандидат и источник OOF/test predictions для `exp-012`.
