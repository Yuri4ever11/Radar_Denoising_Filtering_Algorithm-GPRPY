import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# 生成复杂多样的二维雷达图像
def generate_radar_image(size):
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X ** 2 + Y ** 2)) + 0.5 * np.cos(0.5 * X) - 0.2 * Y
    return Z


# 显示图像和处理前后对比
def show_comparison(original, processed):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='viridis')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='viridis')
    plt.title('Processed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 生成复杂多样的二维雷达图像
size = 200
radar_image = generate_radar_image(size)

# 使用短时傅里叶变换进行时频分析
frequencies, times, spectrogram_data = spectrogram(radar_image, fs=1, nperseg=64, noverlap=32)

# 对时频图谱进行处理（示例中仅为清零低频成分）
processed_spectrogram_data = spectrogram_data.copy()
processed_spectrogram_data[frequencies < 0.2] = 0

# 使用逆短时傅里叶变换获取处理后的图像
processed_radar_image = np.fft.irfft(processed_spectrogram_data, axis=0)

# 显示处理前后对比
show_comparison(radar_image, processed_radar_image)
