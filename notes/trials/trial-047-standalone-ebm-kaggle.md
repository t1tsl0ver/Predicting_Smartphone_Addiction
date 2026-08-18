# trial-047 — Standalone EBM на Kaggle

- Статус: **cancelled by participant before run; no result**
- Среда: Kaggle CPU notebook
- Контроль: standalone LightGBM на тех же folds и числовой матрице

Участник отказался от направления новых семейств моделей до запуска notebook.
Файл сохранён только как историческая заготовка; не запускать и не трактовать
описанный ниже план как активный trial.

## Вопрос

Может ли Explainable Boosting Machine дать конкурентный standalone OOF и более
ортогональные ошибки, чем LightGBM, на текущем non-TE feature set?

## Гипотеза

Аддитивные одномерные функции и автоматически выбранные парные взаимодействия
EBM имеют другую индуктивную структуру, чем обычный GBDT. Четыре engineered
фичи уже раскрывают часть взаимодействий, поэтому EBM может оказаться полезным
даже при более ограниченной структуре.

## Что меняется

Только семейство модели: `ExplainableBoostingClassifier` вместо LightGBM.
Используются стандартные параметры InterpretML `0.7.8`: 14 outer bags,
`interactions='3x'`, LR `0.015`, максимум 50000 rounds и early stopping 100.

## Что остаётся неизменным

- два stratified folds, shuffle, seed 2026;
- 17 смысловых фич, включая четыре победителя `trial-043`;
- target encoding полностью отсутствует;
- две frequency-фичи строятся только по fold-train;
- ROC AUC по fold percentile ranks;
- кандидат и notebook-LightGBM получают одинаковую числовую матрицу.

Сильный project reference exact-LightGBM из `trial-043` также показывается
отдельно: общий OOF `0.9637208762`.

## План проверки

Notebook сохраняет общий и fold-wise OOF, дельту к своему standalone LightGBM,
дельту к project exact-LightGBM, Spearman-корреляцию, paired bootstrap и точное
время каждого fit. До возврата `trial-047-output.zip` результата нет.

## Запуск

Импортировать `kaggle/trial-047-ebm.ipynb`, прикрепить competition data,
оставить CPU, включить Internet и выполнить Run All. Подробности находятся в
`kaggle/README.md`.
