import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


# 模拟生成带有噪声的雷达信号
def generate_radar_signal(size):
    signal = np.zeros((size, size))
    # 在信号中添加模拟的雷达反射信号
    signal[size // 2, size // 2] = 10
    # 添加随机噪声
    noise = np.random.normal(0, 2, (size, size))
    noisy_signal = signal + noise
    return noisy_signal


# 自适应中值滤波函数
def adaptive_median_filter(signal, window_size):
    filtered_signal = np.zeros_like(signal)
    pad = window_size // 2
    for i in range(pad, signal.shape[0] - pad):
        for j in range(pad, signal.shape[1] - pad):
            window = signal[i - pad:i + pad + 1, j - pad:j + pad + 1]
            window_median = np.median(window)
            window_range = np.max(window) - np.min(window)
            signal_value = signal[i, j]
            if signal_value > window_median - window_range and signal_value < window_median + window_range:
                filtered_signal[i, j] = signal_value
            else:
                filtered_signal[i, j] = window_median
    return filtered_signal


# 显示信号和处理前后对比
def show_comparison(original, processed):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='viridis', vmin=0, vmax=10)
    plt.title('Original Signal')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(processed, cmap='viridis', vmin=0, vmax=10)
    plt.title('Processed Signal')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(original - processed, cmap='coolwarm')
    plt.title('Difference')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 生成模拟的雷达信号
size = 100
radar_signal = generate_radar_signal(size)

# 对信号进行自适应中值滤波处理
window_size = 5  # 调整自适应中值滤波的窗口大小，以观察效果
processed_radar = adaptive_median_filter(radar_signal, window_size)

# 显示处理前后对比
show_comparison(radar_signal, processed_radar)

"""
差异图像的作用主要有以下几点：

可视化处理效果：通过差异图像，
您可以直观地看到信号在经过自适应中值滤波处理后发生的变化。
亮的区域表示处理后的信号较亮，暗的区域表示处理后的信号较暗，
这有助于您判断滤波是否在去除噪声的同时保留了信号的特征。

评估滤波效果：差异图像可以用于定量地评估滤波效果。
如果差异图像中明显的亮暗区域较少，说明滤波效果较好，处理后的信号与原始信号相似。
如果差异图像中存在较大的亮暗区域，可能意味着滤波对信号产生了较大的变化。

调整滤波参数：差异图像可以帮助您调整滤波参数。
如果差异图像中存在明显的细节损失，您可以尝试调整滤波参数，如窗口大小等，
以平衡去噪和信号保留之间的关系。
"""