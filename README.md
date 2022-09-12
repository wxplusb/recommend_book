## Решение "Цифровой прорыв 2022. Разработка рекомендательного алгоритма для читателей библиотеки"

### Настройка окружения с помощью miniconda:

```
conda create -n recsys_env python=3.8.12
conda activate recsys_env
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
```


### Воспроизведение решения:
  1. Создать папку data рядом с ноутбуками и файлами, в нее поместить тренировочные данные, sample_solution.csv.
  2. [1_prepro.ipynb](1_prepro.ipynb) - Предобработка данных
  3. [2_train_predict.ipynb](2_train_predict.ipynb) - Тренировка моделей и получение предсказаний.


### License
[MIT license](LICENSE).

