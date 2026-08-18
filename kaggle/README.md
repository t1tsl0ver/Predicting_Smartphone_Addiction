# Kaggle model-family trials

Здесь лежат два независимых OOF-notebook. Они не создают submission и не
используют target encoding:

- `trial-047-ebm.ipynb` — standalone EBM на CPU;
- `trial-048-pyboost.ipynb` — standalone Py-Boost на NVIDIA GPU.

Оба notebook используют одинаковые два `StratifiedKFold`, `seed=2026`, четыре
принятые фичи `trial-043` и самостоятельно обучают одиночный LightGBM-контроль
на тех же строках. Пропуски обозначаются значением `-999`, а
`academic_work_impact` кодируется `No=0`, `Yes=1`; это одинаково для кандидата и
его notebook-контроля. Frequency mapping строится только по train-части fold.

## Как загрузить

1. Открыть Kaggle и перейти в **Code → New Notebook**.
2. В редакторе выбрать **File → Import Notebook** и перетащить нужный `.ipynb`.
3. Справа нажать **Add Input** и прикрепить данные текущего соревнования. Если
   competition data не показываются, создать приватный Kaggle Dataset из
   локального `train.csv` и добавить его через **Your Datasets**.
4. В **Settings** включить **Internet On**, потому что первая ячейка ставит
   зафиксированные версии библиотек.
5. Для EBM оставить **Accelerator: None**. EBM использует CPU.
6. Для Py-Boost обязательно выбрать **Accelerator: GPU**. В первой ячейке
   `nvidia-smi` и CuPy проверят, что NVIDIA действительно доступна.
7. Запустить **Run All**. Для фонового полного запуска можно выбрать
   **Save Version → Save & Run All**. Не запускать две версии одного notebook
   одновременно, чтобы не тратить GPU quota.

Notebook сам найдёт подходящий `train.csv` внутри `/kaggle/input`. Если данные
не прикреплены или выбрана не та таблица, он завершится до обучения с понятной
ошибкой.

## Что скачать после запуска

В правой панели **Output** появится один архив:

```text
trial-047-output.zip
trial-048-output.zip
```

Каждый архив содержит:

- JSON с параметрами, версиями, общим/fold-wise OOF, bootstrap и временем;
- полный OOF CSV кандидата и его LightGBM-контроля;
- importance модели.

Скачать оба ZIP и вернуть их в проект. Только после чтения фактических outputs
trial получают итоговый статус, карточки результатов и отдельные result-коммиты.
