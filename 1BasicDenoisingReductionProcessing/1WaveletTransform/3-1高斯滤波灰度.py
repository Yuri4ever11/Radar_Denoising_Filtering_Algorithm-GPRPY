import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# 创建一个复杂多样的灰度二维雷达图像
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
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original - processed, cmap='coolwarm')
    plt.title('Difference')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 生成复杂多样的灰度二维雷达图像
size = 200
radar_image = generate_radar_image(size)

# 对图像进行高斯滤波处理
sigma = 3  # 调整高斯滤波的标准差，以观察效果
processed_radar = gaussian_filter(radar_image, sigma=sigma)

# 显示处理前后对比
show_comparison(radar_image, processed_radar)
