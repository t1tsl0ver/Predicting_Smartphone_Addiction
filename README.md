# Predicting Smartphone Addiction

Учебный Kaggle-проект по бинарной классификации. По характеристикам использования
смартфона модель оценивает вероятность `addicted_label = 1`. Метрика соревнования —
**ROC AUC**, поэтому в submission нужны вероятности, а не готовые классы 0/1.

## Текущий результат

Новый локальный итоговый эксперимент —
[`exp-002`](notes/experiments/exp-002-max-lightgbm.md); его Kaggle score ожидается:

| Версия | Локальная проверка | Kaggle leaderboard |
|---|---:|---:|
| `exp-000`: CatBoost d7/400 | holdout 0.955008 | 0.95663 |
| `exp-001`: 3-fold rank ensemble | **OOF 0.963154** | **0.96466** |
| `exp-002`: 5-fold high-budget LightGBM | **OOF 0.964047** | pending |

`exp-002` использует только LightGBM и принятые feature-engineering улучшения;
увеличенный бюджет дал `+0.001802` OOF к быстрому `trial-014`. История маленьких
проверок находится в [`notes/TRIALS.md`](notes/TRIALS.md).

Активная быстрая линия дальнейших trial использует чистый LightGBM на 500
деревьев с 5-fold CV. CatBoost-ансамбль сохранён как исторический `exp-001`, но
в новых trial не обучается. Текущий лучший trial этой линии — `trial-014`: OOF
`0.962243`, Kaggle `0.96412`.

## Структура репозитория

```text
data/                    локальные CSV, сами данные игнорируются Git
docs/competition/        сохранённые условия соревнования
notes/                   контекст проекта и правила экспериментов
notes/experiments/       отдельная карточка каждого эксперимента
src/train.py             валидация, обучение и создание submission
src/cv_lightgbm.py       быстрые LGBM-only CV, OOF и optional submission
src/progress_logging.py  общие консольные и файловые логи прогресса
tests/                   быстрые проверки инфраструктурного кода
artifacts/               локальные модели и метрики, игнорируются Git
requirements.txt         зафиксированные версии библиотек
```

Перед продолжением работы прочитай [`notes/CURRENT_STATE.md`](notes/CURRENT_STATE.md).
Подробная логика первого решения записана в карточке эксперимента.

В корне поддерживается один актуальный Kaggle-файл `submission.csv`. Он создаётся
или перезаписывается только по явной просьбе; отдельные файлы с суффиксами
trial/exp не сохраняются.

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

## Логи выполнения

Все скрипты показывают этапы работы в консоли и одновременно сохраняют те же
сообщения в `artifacts/logs/`. Имя файла содержит время запуска, название скрипта
и `run-id`, поэтому логи разных запусков не перезаписываются.

Во время обучения выводятся:

- текущий этап (`START`, `PROGRESS`, `DONE` или `FAILED`);
- номер итерации, процент выполнения и примерный ETA;
- текущие train/validation-метрики;
- активные fold и модель при CV;
- длительность этапов и пути сохранённых файлов.

Частота сообщений настраивается без изменения модели:

```powershell
.\.venv\Scripts\python.exe src\train.py --model catboost --iterations 1000 --log-every 50 --skip-final
.\.venv\Scripts\python.exe src\cv_ensemble.py --catboost-log-every 50 --lightgbm-log-every 200
```

`src/cv_lightgbm.py` по умолчанию не создаёт Kaggle CSV. Submission записывается
только по явной команде с `--submission путь.csv`; метрики, OOF и test predictions
при этом сохраняются для каждого trial независимо.

Последний лог можно наблюдать из второго PowerShell-окна во время работы:

```powershell
$latestLog = Get-ChildItem artifacts\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content $latestLog.FullName -Wait
```

## Как ведутся эксперименты

Маленькие проверки нумеруются `trial-001`, `trial-002`, … и получают отдельные
коммиты. Принятые улучшения периодически объединяются в версии `exp-001`,
`exp-002`, … после получения Kaggle score. Git-теги не используются. До реализации
мы отдельно обсуждаем гипотезу, единственное основное изменение, ожидаемый эффект
и способ проверки. Полный протокол находится в
[`notes/EXPERIMENT_RULES.md`](notes/EXPERIMENT_RULES.md).
