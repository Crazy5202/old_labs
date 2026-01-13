
import numpy as np
import matplotlib.pyplot as plt

# группа - 12; номер по списку - 6
k = 12
l = 15

alpha = 1 / (1 + k) 
shift = l / 2

# создаём N точек на интервале [0, 1]
N = 501
t_grid = np.linspace(0, 1, N)

# интерполяция на интервале [0, 1] для более гладкого графика
def get_value_x(x_array, t): 
    if t <= 0:
        return x_array[0] 
    if t >= 1:
        return x_array[-1] 
    pos = t * (N - 1)
    i_left = int(np.floor(pos)) 
    i_right = min(i_left + 1, N - 1) 
    alpha_ = pos - i_left
    return (1 - alpha_) * x_array[i_left] + alpha_ * x_array[i_right]

# применение сжимающего отображения
def apply_T(x_array):
    new_x = np.zeros_like(x_array)
    # Значения в точках разрыва для обеспечения непрерывности 
    left_boundary = alpha * get_value_x(x_array, 1) - shift # t = 1/3 
    right_boundary = alpha * get_value_x(x_array, 0) + shift # t = 2/3 
    for i, t in enumerate(t_grid):
        if t <= 1/3:
            val = get_value_x(x_array, 3*t) 
            new_x[i] = alpha * val - shift
        elif t >= 2/3:
            val = get_value_x(x_array, 3*(t - 2/3)) 
            new_x[i] = alpha * val + shift
        else: # Ломаная на [1/3; 2/3] через точки (4/9; 1) и (5/9; -1) 
            if t <= 4 / 9:
            # Линейный участок от (1/3; left_boundary) до (4/9; 1) 
                t_normalized = (t - 1 / 3) / (4 / 9 - 1 / 3)
                new_x[i] = (1 - t_normalized) * left_boundary + t_normalized * 1 
            elif t <= 5 / 9:
                # Линейный участок от (4/9; 1) до (5/9; -1) 
                t_normalized = (t - 4 / 9) / (5 / 9 - 4 / 9)
                new_x[i] = (1 - t_normalized) * 1 + t_normalized * (-1) 
            else:
                # Линейный участок от (5/9; -1) до (2/3; right_boundary) 
                t_normalized = (t - 5 / 9) / (2 / 3 - 5 / 9)
                new_x[i] = (1 - t_normalized) * (-1) + t_normalized * right_boundary 
    return new_x

    # рассчёт нормы
def uniform_norm(x1, x2):
    return np.max(np.abs(x1 - x2))

# Начальное приближение x0(t) = t 
x0_array = t_grid.copy()

iterates = [x0_array] 
current_x = x0_array

# Итерационный процесс с остановкой по точности eps 
eps = 1e-3
iters = np.ceil(np.log((eps * (1 - alpha) / uniform_norm(x0_array, apply_T(x0_array)))) / np.log(alpha)).astype(int)


for n in range(iters):
    next_x = apply_T(current_x) 
    iterates.append(next_x) 
    current_x = next_x

# Визуализация всех итераций на отдельных подграфиках
fig, axes = plt.subplots(1, len(iterates), figsize=(15, 4), sharex=True, sharey=True)

for i, ax in enumerate(axes):
    ax.plot(t_grid, iterates[i], color='blue', label=f'$x_{i}(t)$') 
    ax.set_title(f'Итерация {i}')
 
ax.grid(True) 
ax.legend()


plt.suptitle(f"Итерации метода сжимающих отображений для eps = {eps}") 
plt.show()
