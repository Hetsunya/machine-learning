from PIL import Image
import numpy as np
import pandas as pd
from neural_network import  create_neural_network, train_neural_network, split_data, Neuron, evaluate_neural_network

# Шаг 1: Загрузка изображения и получение массива цветов пикселей
image = Image.open('kotik.jpg')
pixel_colors = np.array(image)

# Шаг 2: Выбор цвета, который вы хотите выделить, и выделение соответствующих участков
target_color = np.array([10, 106, 69])
tolerance = 40

def is_target_color(color):
    return np.all(np.abs(color - target_color) < tolerance)

mask_target_color = np.array([[is_target_color(color) for color in row] for row in pixel_colors])
pixels_with_target_color = pixel_colors[mask_target_color]
pixels_without_target_color = pixel_colors[~mask_target_color]

# Шаг 3: Создание датасета для обучения
df_with_target_color = pd.DataFrame(pixels_with_target_color.reshape(-1, 3), columns=['R', 'G', 'B'])
df_with_target_color['our color'] = 1

df_without_target_color = pd.DataFrame(pixels_without_target_color.reshape(-1, 3), columns=['R', 'G', 'B'])
df_without_target_color['our color'] = 0

# Объединение датасетов
training_data = pd.concat([df_with_target_color, df_without_target_color], ignore_index=True)

# Вывод первых нескольких строк датасета
print(training_data)

# Нормализуем данные
training_data[['R', 'G', 'B']] = training_data[['R', 'G', 'B']] / 255.0

# Преобразуем датасет в формат, подходящий для обучения
training_data_array = np.array(training_data[['R', 'G', 'B']])
target_labels = np.array(training_data['our color'])

# Разделим данные на обучающую и тестовую выборки
train_data, test_data = split_data(list(zip(training_data_array, target_labels)), train_fraction=0.8)

# Создаем и обучаем нейронную сеть
num_inputs_per_neuron = 3  # Входы: R, G, B
num_neurons = 1  # Один нейрон, так как у нас бинарная классификация

neural_net = create_neural_network(num_neurons, num_inputs_per_neuron)
train_neural_network(neural_net, train_data, learning_rate=0.00001, epochs=1)

# Используйте эту функцию после обучения нейронной сети
evaluate_neural_network(neural_net, test_data)
evaluate_neural_network(neural_net, train_data)


# ДРУГАЯ КАРТИНКА LOL



# # Шаг 1: Загрузка тестирующей картинки и получение массива цветов пикселей
# test_image = Image.open('test.jpg')
# test_pixel_colors = np.array(test_image)
#
# # Шаг 2: Выделение участков цвета
# test_mask_target_color = np.array([[is_target_color(color) for color in row] for row in test_pixel_colors])
# test_pixels_with_target_color = test_pixel_colors[test_mask_target_color]
# test_pixels_without_target_color = test_pixel_colors[~test_mask_target_color]
#
# # Шаг 3: Создание датасета для тестирования
# test_df_with_target_color = pd.DataFrame(test_pixels_with_target_color.reshape(-1, 3), columns=['R', 'G', 'B'])
# test_df_with_target_color['our color'] = 1
#
# test_df_without_target_color = pd.DataFrame(test_pixels_without_target_color.reshape(-1, 3), columns=['R', 'G', 'B'])
# test_df_without_target_color['our color'] = 0
#
# # Объединение датасетов для тестирования
# test_data = pd.concat([test_df_with_target_color, test_df_without_target_color], ignore_index=True)
#
# # Вывод первых нескольких строк тестового датасета
# print(test_data.head())
#
# # Нормализация данных
# test_data[['R', 'G', 'B']] = test_data[['R', 'G', 'B']] / 255.0
#
# # Преобразование датасета в формат, подходящий для тестирования
# test_data_array = np.array(test_data[['R', 'G', 'B']])
# test_target_labels = np.array(test_data['our color'])
#
# # Используйте эту функцию для оценки нейронной сети на тестовых данных
# evaluate_neural_network(neural_net, list(zip(test_data_array, test_target_labels)))