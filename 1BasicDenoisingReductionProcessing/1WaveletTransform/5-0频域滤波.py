import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2


# 创建一个复杂多样的二维雷达图像
def generate_radar_image(size):
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X ** 2 + Y ** 2)) + 0.5 * np.cos(0.5 * X) - 0.2 * Y

    return Z


# 显示图像和处理前后对比
def show_comparison(original, processed):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='viridis')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(processed, cmap='viridis')
    plt.title('Processed Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original - processed, cmap='coolwarm')
    plt.title('Difference')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 生成复杂多样的二维雷达图像
size = 200
radar_image = generate_radar_image(size)

# 对图像进行频域滤波（低通滤波）处理
cutoff_frequency = 0.1  # 调整截止频率，以观察效果
frequency_image = fft2(radar_image)
filtered_frequency_image = frequency_image.copy()
filtered_frequency_image[
    np.abs(frequency_image) > cutoff_frequency * np.max(np.abs(frequency_image))
    ] = 0
processed_radar = np.real(ifft2(filtered_frequency_image))

# 显示处理前后对比
show_comparison(radar_image, processed_radar)
