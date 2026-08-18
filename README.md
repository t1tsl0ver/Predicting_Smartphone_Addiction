# Predicting Smartphone Addiction

Учебный Kaggle-проект по бинарной классификации. По характеристикам использования
смартфона модель оценивает вероятность `addicted_label = 1`. Метрика соревнования —
**ROC AUC**, поэтому в submission нужны вероятности, а не готовые классы 0/1.

## Текущий результат

Новый локальный итоговый эксперимент —
[`exp-005`](notes/experiments/exp-005-exp003-exp002-rank-blend.md):

| Версия | Локальная проверка | Kaggle leaderboard |
|---|---:|---:|
| `exp-000`: CatBoost d7/400 | holdout 0.955008 | 0.95663 |
| `exp-001`: 3-fold rank ensemble | **OOF 0.963154** | **0.96466** |
| `exp-002`: 5-fold high-budget LightGBM | **OOF 0.964047** | pending |
| `exp-003`: exact-value LightGBM, LR 0.01 | **OOF 0.965044** | **0.96681** |
| `exp-004`: high-capacity exact-value LightGBM | **OOF 0.964587** | **0.96668** |
| `exp-005`: 60/40 rank blend `exp-003` + `exp-002` | **OOF 0.966874** | **0.96827** |

`exp-003` использует принятые feature-engineering улучшения и categorical-копии
точных числовых значений. Меньший learning rate и fold early stopping дали
`+0.000997` OOF к `exp-002`. История маленьких проверок находится в
[`notes/TRIALS.md`](notes/TRIALS.md).

`exp-004` перенёс exact-value encoding в более ёмкую конфигурацию `exp-002`.
Она улучшила `exp-002`, но проиграла `exp-003` на всех 5 folds, поэтому локальным
лидером среди одиночных моделей остаётся `exp-003`. `exp-005` смешал ranks
`exp-003` и `exp-002` в пропорции 60/40 и стал новым общим локальным лидером с
OOF `0.966874` и новым leaderboard-лидером со score `0.96827`.

Серия `trial-008`–`trial-021` использовала чистый LightGBM с 5-fold CV.
`trial-019` добавил categorical-копии точных числовых значений и
повысил OOF до `0.964025` (`+0.001782` к `trial-014`), выиграв 5/5 folds.
`trial-020` усилил categorical smoothing, но немного снизил OOF до `0.963977`,
а `trial-021` сократил бюджет до 250 деревьев и повысил OOF до `0.964279`,
выиграв 5/5 folds. Активным контролем стал `trial-021`.
До `trial-022` CatBoost-ансамбль сохранялся только как исторический `exp-001`.

`trial-022` вернул CatBoost в активную линию: при одинаковых 200 деревьях он
выиграл у LightGBM на всех 5 folds (`0.965049` против `0.964293`). В
`trial-023` на быстром 2-fold протоколе exact/base 60/40 blend заметно помог
LightGBM (`+0.001713`), но навредил CatBoost (`-0.001895`), потому что CatBoost
без exact-категорий оказался слишком слабым. Следующие trial используют 2 folds;
перспективный кандидат перед финальной версией подтверждается на 5 folds.

`trial-024` удалил наименее важную общую exact-фичу `gaming_hours_exact`.
CatBoost улучшился на `+0.000073`, выиграл 2/2 folds и ускорился примерно на 8%;
для его быстрой screening-линии ablation принята. У LightGBM blend дельта всего
`+0.000018`, один fold отрицательный и bootstrap пересекает ноль, поэтому его
feature set не меняется. Канонический быстрый конфиг, включая обязательные два
folds, хранится в `src/trial_config.py`.

`trial-025` начал серию строго одиночных LightGBM feature-проверок. Фича
`notifications_per_screen_hour` дала только `+0.000012` к exact/base blend,
проиграла на одном из двух folds, а paired interval пересёк ноль; кандидат
отклонён. Для серии добавлен общий `src/cv_feature_trial.py`. Комбинации и target
encoding не запускаются до завершения и ранжирования одиночных кандидатов.

`trial-026` проверил `app_opens_per_screen_hour`: blend delta `-0.000023`, один
fold отрицательный, paired interval пересёк ноль. Второй ratio-кандидат также
отклонён; оба отношения сильнее ухудшили base-компонент, чем exact.

`trial-027` добавил `app_opens_per_notification` и получил фактически нулевую
дельту `+0.000001`: folds разошлись по знаку, bootstrap симметрично пересёк ноль.
Exact немного выиграл, но base снова ухудшился; кандидат отклонён.

`trial-028` нашёл первого победителя одиночной серии: доля развлечений
`(social + gaming) / daily` повысила LightGBM blend OOF на `+0.000118`, выиграла
2/2 folds, paired interval полностью выше нуля. Фича принята для screening, но
не объединяется с другими до завершения рейтинга и отдельного решения.

`trial-029` проверил долю work/study в общем экранном времени и стал новым
лидером серии: `+0.000452` OOF, победа 2/2 folds, paired interval полностью выше
нуля. Exact-компонент вырос на `+0.000806`, base остался почти неизменным.

`trial-030` изолированно проверил `weekend - daily`: blend снизился на
`-0.000033`, кандидат проиграл 2/2 folds и был отклонён.

`trial-031` добавил число пропусков: точечная delta `+0.000028`, но folds
разошлись по знаку и paired interval пересёк ноль. Кандидат отклонён как
неподтверждённый.

`trial-032` начал fold-safe frequency-серию с частоты точного `age`. Blend
снизился на `-0.000025`, кандидат отклонён. Frequency mapping строится только по
train-части каждого fold; target не используется.

`trial-033` проверил частоту точного daily screen time и дал слабый положительный
сигнал: `+0.000048`, оба folds формально положительны, paired lower bound едва
выше нуля. Кандидат принят для screening с пометкой «пограничный».

`trial-034` проверил частоту social-media hours и уверенно проиграл: blend
`-0.000150`, 2/2 отрицательных folds, paired interval полностью ниже нуля.

`trial-035` проверил frequency gaming hours и был отклонён: blend `-0.000012`,
folds разошлись по знаку, оба компонента ухудшились.

`trial-036` проверил frequency work/study hours: `+0.000016`, mixed folds и
интервал через ноль; кандидат отклонён.

`trial-037` проверил frequency sleep hours и был устойчиво отклонён: blend
`-0.000041`, 2/2 проигранных folds, paired interval ниже нуля.

## Структура репозитория

```text
data/                    локальные CSV, сами данные игнорируются Git
docs/competition/        сохранённые условия соревнования
notes/                   контекст проекта и правила экспериментов
notes/experiments/       отдельная карточка каждого эксперимента
src/train.py             валидация, обучение и создание submission
src/cv_lightgbm.py       быстрые LGBM-only CV, OOF и optional submission
src/cv_family_blends.py  2-fold exact/base blends CatBoost и LightGBM
src/cv_drop_exact_feature.py парная ablation exact-фичи против cached OOF
src/cv_feature_trial.py  одиночные LightGBM feature-trial против cached blend
src/trial_config.py      канонические параметры быстрых 2-fold trial
src/blend_cv.py          проверенный OOF/test rank blend и submission
src/compare_oof.py       paired bootstrap для сравнения двух OOF-наборов
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
