# trial-023 — Двухфолдовые CatBoost- и LightGBM-family blends

- Статус: **completed; LightGBM blend accepted as a screening signal,
  CatBoost 60/40 blend rejected**
- Дата: 2026-08-17
- Контроль внутри каждого семейства: exact-value модель из `trial-022`
- Режим: быстрый 2-fold trial, не финальная 5-fold проверка

## Вопрос

Если одиночный CatBoost превзошёл одиночный LightGBM в `trial-022`, сохранится
ли его преимущество после добавления внутри каждого семейства второй модели без
exact-value категорий, аналогично идее успешного LightGBM blend `exp-005`?

## Гипотеза

Exact- и base-представления могут делать разные ошибки. Фиксированный rank blend
двух представлений способен превзойти сильную exact-модель. Если эффект
одинаково доступен обоим семействам, CatBoost-family blend должен остаться выше
LightGBM-family blend.

## Заранее заданный дизайн и критерий

- `StratifiedKFold(n_splits=2, shuffle=True, random_state=2026)`;
- одинаковые folds для всех моделей;
- параметры и бюджет моделей полностью взяты из `trial-022`;
- exact-вариант: 22 принятых признака, включая девять `_exact`-копий;
- base-вариант: те же признаки без девяти `_exact`-копий, всего 13;
- фиксированный blend: 60% exact fold rank + 40% base fold rank;
- никаких test predictions и submission.

Family blend считается полезным screening-сигналом, если он превосходит exact-
компонент в общем OOF и на обоих folds. Между семействами сравниваются их
фиксированные 60/40 blends. Вес не подбирается по основному результату trial.

## Параметры

```text
LightGBM: iterations=200, learning_rate=0.10, num_leaves=31,
          min_child_samples=20, cat_smooth=10, reg_alpha=0, reg_lambda=0
CatBoost: iterations=200, learning_rate=0.08, depth=8, l2_leaf_reg=5
Shared:   early_stopping=False, folds=2, seed=2026
Blend:    60% exact fold rank + 40% base fold rank
```

## Основной результат

| Семейство | Exact rank OOF | Base rank OOF | 60/40 blend OOF | Blend − exact |
|---|---:|---:|---:|---:|
| LightGBM | 0.9628257006 | 0.9594471550 | **0.9645386674** | **+0.0017129668** |
| CatBoost | **0.9645158783** | 0.9510225236 | 0.9626206437 | **-0.0018952346** |

При одинаковом весе 60/40 LightGBM-family blend выше CatBoost-family blend на
`0.0019180237`.

### Сравнение по folds

| Fold | LGBM exact | LGBM blend | Delta | Cat exact | Cat blend | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9628891419 | **0.9646319882** | **+0.0017428463** | **0.9643967987** | 0.9625559584 | **-0.0018408403** |
| 2 | 0.9627622586 | **0.9644457175** | **+0.0016834589** | **0.9646349590** | 0.9626849702 | **-0.0019499888** |

Направления полностью повторились на обоих folds.

### Paired bootstrap основного фиксированного сравнения

Использовано 100 стратифицированных повторов размера test (`296302`). Интервалы
оценивают row-sampling uncertainty и не покрывают смену folds или обучение.

| Сравнение | Mean delta | 95% interval | Positive repeats |
|---|---:|---:|---:|
| Cat blend − Cat exact | -0.0018987 | [-0.0020100, -0.0017647] | 0/100 |
| LGBM blend − LGBM exact | +0.0017123 | [+0.0015754, +0.0018258] | 100/100 |
| Cat blend − LGBM blend | -0.0019186 | [-0.0020819, -0.0017263] | 0/100 |

## Post-hoc диагностика весов

После основного фиксированного результата по сохранённым OOF выполнен
диагностический sweep exact-веса от 0 до 1 с шагом 0.05. Он не является
заранее заданным основным результатом.

- LightGBM: общий максимум при 65/35, OOF `0.9645436314`. Первый fold выбрал
  60/40, второй — 65/35. Cross-fold применение весов дало `0.9645382647`, то
  есть практически повторило заранее заданные 60/40.
- CatBoost: общий максимум и оба отдельных fold-максимума получены при 95/5,
  OOF `0.9645334873`. Прирост к exact равен всего `+0.0000176090`.
- Bootstrap CatBoost 95/5 минус CatBoost exact дал интервал
  `[-0.0000006, +0.0000318]`: ноль внутри, поэтому улучшение не доказано.
- CatBoost 95/5 и cross-fold LightGBM blend практически равны. Bootstrap
  CatBoost минус LightGBM дал `[-0.0001687, +0.0001793]`.

Корреляция exact/base ranks почти одинакова у семейств (`0.96903` у LightGBM и
`0.96898` у CatBoost). Различие объясняется не большей diversity LightGBM, а
уровнем второго компонента: base LightGBM отстаёт от exact примерно на
`0.00338`, а base CatBoost — на `0.01349`. Вес 40% для base CatBoost слишком
велик.

## Время

- два exact CatBoost fit: около `7:29`;
- два base CatBoost fit: около `1:42`;
- четыре LightGBM fit вместе: около `0:37`;
- весь model/CV цикл: `9:50`;
- три bootstrap-сравнения: `0:39`;
- полный запуск с подготовкой и сохранением: `10:43`.

Для обычного будущего trial с одной exact CatBoost-конфигурацией два folds
должны занимать около семи с половиной минут на этой машине против `22:42`
обучения пяти exact CatBoost в `trial-022`. Абсолютные 2-fold и 5-fold AUC нельзя
сравнивать напрямую, потому что в 2-fold модель обучается только на половине
строк. Двухфолдовый режим используется для быстрых парных trial; кандидат для
финальной версии подтверждается на 5 folds.

## Решение

- Фиксированный CatBoost 60/40 blend отклонён: он сильно и одинаково проиграл
  exact CatBoost на обоих folds.
- LightGBM 60/40 blend принят как устойчивый screening-сигнал: он заметно
  улучшил exact LightGBM на обоих folds.
- Post-hoc CatBoost 95/5 не принимается как улучшение: дельта мала, интервал
  пересекает ноль, а вес найден после просмотра OOF.
- На trial-бюджете сильный exact CatBoost и LightGBM-family blend фактически
  равны. Для следующей CatBoost-гипотезы контролем остаётся exact CatBoost без
  обязательного base-компонента.
- Submission не создавался.

## Команда воспроизведения

```powershell
.\.venv\Scripts\python.exe -u src\cv_family_blends.py --run-id trial-023 --folds 2 --iterations 200 --seed 2026 --exact-weight 0.60 --log-every 50 --bootstrap-repeats 100
```

Артефакты:

- `artifacts/metrics_trial-023_family_blends.json`;
- `artifacts/oof_predictions_trial-023.csv`;
- `artifacts/logs/20260817T172847431180+0300_cv-family-blends_trial-023.log`.
