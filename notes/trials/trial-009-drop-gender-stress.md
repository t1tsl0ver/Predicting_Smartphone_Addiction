# trial-009 — Drop gender and stress level

- Статус: **accepted; positive overall OOF, won 3/5 folds**
- Дата: 2026-08-14
- Контроль: чистый LightGBM/500 с 5-fold CV из `trial-008`

## Вопрос

Улучшит ли чистый LightGBM удаление двух признаков, которые практически не
используются моделью: `gender` и `stress_level`?

## Гипотеза

В среднем по пяти моделям `trial-008` gain importance равен примерно `284` для
`gender` и `213` для `stress_level`, тогда как важности сильных признаков достигают
десятков и сотен миллионов. Удаление этих колонок может убрать шум без потери
полезного сигнала.

## Единственное основное изменение

```text
drop_features=[gender, stress_level]
```

`academic_work_impact` остаётся: его gain importance в `trial-008` заметно выше.
Числовые `age`, `sleep_hours` и `notifications_per_day` также остаются, потому что
слабая линейная корреляция не исключает полезных порогов и взаимодействий.

## Что остаётся неизменным

```text
model=LGBMClassifier
n_estimators=500
learning_rate=0.1
num_leaves=31
min_child_samples=20
reg_alpha=0.0
reg_lambda=0.0
folds=5
shuffle=True
random_state=2026
```

## План проверки

- использовать ровно те же folds, что в `trial-008`;
- сравнить validation AUC на каждом fold;
- сравнить общий raw и fold-rank OOF AUC;
- принять удаление только при устойчивом улучшении, а не по одной importance.

## Команда

```powershell
.\.venv\Scripts\python.exe src\cv_lightgbm.py --run-id trial-009 --drop-features gender stress_level --submission submission_trial-009.csv
```

## Результаты

| Fold | `trial-008` | `trial-009` | Δ удаления |
|---:|---:|---:|---:|
| 1 | 0.9611075 | 0.9598403 | -0.0012672 |
| 2 | 0.9607909 | 0.9623399 | +0.0015489 |
| 3 | 0.9622291 | 0.9624325 | +0.0002034 |
| 4 | 0.9618350 | 0.9618107 | -0.0000243 |
| 5 | 0.9594925 | 0.9604214 | +0.0009290 |

```text
control raw OOF AUC:      0.9610880302
trial raw OOF AUC:        0.9613711622
raw OOF delta:           +0.0002831320

control fold-rank OOF:    0.9610914869
trial fold-rank OOF:      0.9613690723
fold-rank delta:         +0.0002775855

fold wins:                3/5
selected prediction:      raw_probability
elapsed CV time:          90.29 seconds
```

## Вывод

Удаление принято: общий OOF вырос примерно на `+0.00028`, положительная разница
получена на трёх folds из пяти, а четвёртый fold практически равен контролю.
Результат не идеально равномерен из-за заметного падения на fold 1, поэтому
Kaggle score остаётся полезной внешней проверкой.

Submission:

```text
file=submission_trial-009.csv
rows=296302
sha256=46506569990da4f6e3f4cf08b182c11cd850996b1c1b2dcd13fe1119dd9377a9
```
