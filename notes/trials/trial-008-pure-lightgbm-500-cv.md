# trial-008 — Pure LightGBM 500-tree CV

- Статус: **completed; adopted as the new fast baseline**
- Дата: 2026-08-14
- Контроль: LightGBM из `trial-007`

## Вопрос

Какое качество даст быстрый чистый LightGBM без CatBoost, если использовать
простую конфигурацию на 500 деревьев и усреднение по CV folds?

## Гипотеза

По внешнему ориентиру сильный результат может достигаться одним LightGBM с CV.
Отказ от CatBoost должен резко сократить время, а 5-fold обучение позволит каждой
модели видеть 80% train и даст более устойчивое усреднение test predictions.

## Проверяемая конфигурация

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
features=12 исходных признаков без id
```

Это screening полной простой конфигурации, а не абляция одного параметра: точные
параметры внешнего решения неизвестны. CatBoost не обучается.

## План проверки

- получить OOF predictions для каждой train-строки;
- сравнить общий AUC raw probabilities и процентных ranks внутри folds;
- заранее выбрать для submission вариант с более высоким OOF AUC;
- сохранить fold-метрики, средние test predictions, submission и SHA-256;
- сравнить время и OOF с LightGBM из `trial-007` (`0.9630794609` fold-rank OOF).

## Команда

```powershell
.\.venv\Scripts\python.exe src\cv_lightgbm.py --run-id trial-008
```

## Результат

```text
raw probability OOF AUC: 0.9610880302
fold-rank OOF AUC:       0.9610914869
selected prediction:     fold_rank
elapsed CV time:         87.59 seconds
total script time:       about 1 minute 32 seconds
submission SHA-256:      f5c2b6b201d4a5149a1baf717150edc7024aeec646562259b3e844f0422f6758
```

Чистый LightGBM/500 слабее LightGBM/3000 из `trial-007` локально примерно на
`0.001988` OOF AUC, но выполняется существенно быстрее и принят участником как
новая база для следующих быстрых trial. CatBoost в следующих trial не используется.
