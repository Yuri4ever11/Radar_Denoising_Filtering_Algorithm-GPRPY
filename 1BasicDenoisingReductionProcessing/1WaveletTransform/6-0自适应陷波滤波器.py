import numpy as np
import matplotlib.pyplot as plt


# 生成带噪声和干扰的二维图像（示例）
def generate_noisy_image(size, interference_freq, amplitude_interference):
    x = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, x)
    noise = np.random.normal(0, 0.1, (size, size))
    image = np.sin(2 * np.pi * X * 5) + amplitude_interference * np.sin(2 * np.pi * X * interference_freq) + noise
    return image


# 自适应陷波滤波器
def adaptive_notch_filter_2D(image, interference_freq, step_size, order):
    filtered_image = np.zeros_like(image)
    weights = np.zeros((order + 1, order + 1))

    for i in range(order, image.shape[0]):
        for j in range(order, image.shape[1]):
            input_matrix = image[i - order:i + 1, j - order:j + 1]  # 输入矩阵，包括过去的像素值
            input_vector = input_matrix.flatten()  # 将矩阵展开成向量
            output = np.dot(weights.flatten(), input_vector)  # 滤波器输出

            # 更新权重（使用LMS算法）
            error = image[i, j] - output
            weights = weights + step_size * error * input_matrix

            filtered_image[i, j] = output

    return filtered_image


# 生成带噪声和干扰的二维图像
size = 100
interference_freq = 25  # 干扰频率
amplitude_interference = 0.5  # 干扰幅度
image = generate_noisy_image(size, interference_freq, amplitude_interference)

# 自适应陷波滤波参数
step_size = 0.01  # 步长，用于权重更新
order = 5  # 滤波器阶数

# 对图像进行自适应陷波滤波处理
filtered_image = adaptive_notch_filter_2D(image, interference_freq, step_size, order)

# 显示图像和处理前后对比
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='viridis')
plt.title('Original Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='viridis')
plt.title('Filtered Image')
plt.colorbar()

plt.tight_layout()
plt.show()

"""
干扰频率和干扰幅度：在示例中，干扰频率和干扰幅度会影响图像中的干扰程度。
您可以尝试将这些参数调整得更大，以便更明显地看到滤波效果。

步长（Step Size）：适当调整步长可以影响滤波器的收敛速度和稳定性。
如果步长太小，滤波器可能收敛得较慢；如果步长太大，滤波器可能不稳定。您可以尝试不同的步长值。

滤波器阶数（Order）：滤波器阶数影响了滤波器的自适应能力。
较高的阶数可能会更好地适应复杂的信号，但也可能导致过拟合。您可以尝试不同的滤波器阶数。

噪声水平：在生成带噪声的图像时，噪声水平也会影响滤波效果。
增加噪声水平可能会更明显地显示滤波效果。

如果您在实际应用中遇到类似的问题，建议先对参数进行调整，然后根据结果进行进一步优化。
此外，对于更复杂的信号和图像处理，您可能需要更高级的自适应滤波方法或其他信号处理技术。
"""