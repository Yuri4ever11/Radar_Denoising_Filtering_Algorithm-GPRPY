import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2


# 创建一个复杂的二维灰度雷达信号图像（示例）
def generate_radar_signal(size):
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    signal = np.sin(np.sqrt(X ** 2 + Y ** 2)) + 0.5 * np.cos(0.5 * X) - 0.2 * Y
    return signal


# 显示信号和处理前后对比
def show_comparison(original, processed):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Signal')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(processed, cmap='gray')
    plt.title('Processed Signal')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original - processed, cmap='coolwarm')
    plt.title('Difference')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 生成复杂的二维灰度雷达信号图像
size = 200
radar_signal = generate_radar_signal(size)

# 对信号进行频域滤波（低通滤波）处理
cutoff_frequency = 0.1  # 调整截止频率，以观察效果
frequency_signal = fft2(radar_signal)
filtered_frequency_signal = frequency_signal.copy()
filtered_frequency_signal[
    np.abs(frequency_signal) > cutoff_frequency * np.max(np.abs(frequency_signal))
    ] = 0
processed_radar_signal = np.real(ifft2(filtered_frequency_signal))

# 显示处理前后对比
show_comparison(radar_signal, processed_radar_signal)
