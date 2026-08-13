# Predicting Smartphone Addiction

Учебный Kaggle-проект по бинарной классификации. По характеристикам использования
смартфона модель оценивает вероятность `addicted_label = 1`. Метрика соревнования —
**ROC AUC**, поэтому в submission нужны вероятности, а не готовые классы 0/1.

## Текущий результат

Эксперимент [`exp-000`](notes/experiments/exp-000-catboost-baseline.md):

| Модель | Локальный ROC AUC | Kaggle leaderboard |
|---|---:|---:|
| Logistic Regression | 0.910275 | — |
| CatBoost, 400 деревьев | **0.955008** | **0.95663** |

Локальная оценка получена на фиксированном стратифицированном holdout: 20% train,
`random_state=42`. Результат Kaggle сообщён участником после загрузки submission.

Текущий кандидат после `trial-007`: 3-fold CV rank ensemble из 15% CatBoost
d8/1000 и 85% LightGBM. Его OOF ROC AUC — `0.963154`; новый Kaggle score ещё не
получен. Актуальная история маленьких проверок находится в
[`notes/TRIALS.md`](notes/TRIALS.md).

## Структура репозитория

```text
data/                    локальные CSV, сами данные игнорируются Git
docs/competition/        сохранённые условия соревнования
notes/                   контекст проекта и правила экспериментов
notes/experiments/       отдельная карточка каждого эксперимента
src/train.py             валидация, обучение и создание submission
artifacts/               локальные модели и метрики, игнорируются Git
requirements.txt         зафиксированные версии библиотек
```

Перед продолжением работы прочитай [`notes/CURRENT_STATE.md`](notes/CURRENT_STATE.md).
Подробная логика первого решения записана в карточке эксперимента.

## Данные

Скачанные с Kaggle файлы нужно положить в `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

В текущем датасете:

- train: 691 369 строк;
- test: 296 302 строки;
- 9 числовых и 3 категориальных признака;
- в признаках есть примерно 4–20% пропусков;
- доля положительного класса в train — 70.94%.

Данные и полученные модели не хранятся в GitHub.

## Установка на Windows

Нужен Python 3.12. В PowerShell из корня репозитория:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Если команда `py` недоступна, укажи полный путь к установленному `python.exe` при
создании `.venv`.

## Запуск

Быстрый линейный baseline без финального submission:

```powershell
.\.venv\Scripts\python.exe src\train.py --model logistic --skip-final
```

Проверка CatBoost на holdout без финального обучения:

```powershell
.\.venv\Scripts\python.exe src\train.py --model catboost --iterations 400 --skip-final
```

Валидация, обучение на всём train и создание `submission.csv`:

```powershell
.\.venv\Scripts\python.exe src\train.py --model catboost --iterations 400
```

Если валидация уже выполнена, можно обучить только финальную модель:

```powershell
.\.venv\Scripts\python.exe src\train.py --model catboost --iterations 400 --final-only
```

## Как ведутся эксперименты

Маленькие проверки нумеруются `trial-001`, `trial-002`, … и получают отдельные
коммиты. Принятые улучшения периодически объединяются в версии `exp-001`,
`exp-002`, … после получения Kaggle score. Git-теги не используются. До реализации
мы отдельно обсуждаем гипотезу, единственное основное изменение, ожидаемый эффект
и способ проверки. Полный протокол находится в
[`notes/EXPERIMENT_RULES.md`](notes/EXPERIMENT_RULES.md).
