# trial-048 — Standalone Py-Boost на Kaggle GPU

- Статус: **ready for Kaggle GPU; not run yet**
- Среда: Kaggle NVIDIA GPU notebook
- Контроль: standalone LightGBM на тех же folds и числовой матрице

## Вопрос

Может ли другая GPU-реализация GBDT превзойти или полезно разнообразить
standalone LightGBM на текущем non-TE feature set?

## Гипотеза

Py-Boost близок к LightGBM по семейству, поэтому сильная ортогональность не
ожидается. Однако другой алгоритм quantization и построения depth-wise деревьев
может дать отличающееся ранжирование и конкурентный OOF при небольшой цене GPU.

## Что меняется

Только семейство модели: `py-boost==0.5.2`, binary loss `bce`, metric `auc`,
200 деревьев, LR `0.10`, depth 6, минимум 20 строк в листе, L2 `1`, 255 bins,
без early stopping. Это отдельный standalone trial, не blend.

## Что остаётся неизменным

- два stratified folds, shuffle, seed 2026;
- 17 смысловых фич, включая четыре победителя `trial-043`;
- target encoding полностью отсутствует;
- две frequency-фичи строятся только по fold-train;
- ROC AUC по fold percentile ranks;
- кандидат и notebook-LightGBM получают одинаковую числовую матрицу.

Сильный project reference exact-LightGBM из `trial-043` показывается отдельно:
общий OOF `0.9637208762`.

## План проверки

Notebook сначала проверяет NVIDIA через `nvidia-smi` и CuPy, затем сохраняет
общий/fold-wise OOF, дельты, Spearman, paired bootstrap и время каждого fit.
До возврата `trial-048-output.zip` результата нет.

## Запуск

Импортировать `kaggle/trial-048-pyboost.ipynb`, прикрепить competition data,
выбрать GPU, включить Internet и выполнить Run All. Подробности находятся в
`kaggle/README.md`.
