import numpy as np
from matplotlib import pyplot as plt

# Точность поиска
precision = 1e-4


# Разбиение [0;1] на точки для вычислений
split = 600
x = np.linspace(0, 1, split)
array = x.copy()

# Функция получения индекса массива из значения t
def get_index(val) -> int:
    return int(val*(split-1))

# Функция сжатия
def compression(cur_array) -> list:
    new_array = np.zeros_like(cur_array) # Создание нового массива для заполнения
    for t in x: # Проведение вычислений по каждому t из разбиения [0;1]
        if t >= 0 and t <= 1/3:
            new_array[get_index(t)] = -1/13 * cur_array[get_index(3 * t)]
        if t > 1/3 and t < 2/3:
            new_array[get_index(t)] = -1/13 * (1 + cur_array[-1] - np.cos(6 * np.pi * (t - 1/3)))
        if t >= 2/3 and t <= 1: # В конце 
            new_array[get_index(t)] = -1/13 * cur_array[get_index(3 * t - 2)] + 1/13 * (cur_array[0] - cur_array[-1])
    return new_array

# Функция расстояния в ЛП
def distance(array1, array2):
    return max(abs(array1 - array2))

# Выбор числа итераций для заданной точности
num_iters = int(np.log(precision*(1-1/13)/distance(array, compression(array)))/np.log(1/13)) + 1

# Учитывание "нулевую" итерацию
num_iters += 1

# Создание области для отрисовки графиков
fig, axes = plt.subplots(1, num_iters, figsize=(5*num_iters, 5))
fig.suptitle(f'Графики сжимающего отображения для ε = {precision}', fontsize=16)

# Построение графиков
for i in range(num_iters):
    axes[i].plot(x, array)
    axes[i].set_title(f'Итерация {i}')
    array = compression(array) # Проведение сжатия

# Отображение графиков
plt.tight_layout()
plt.show()