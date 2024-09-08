import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# 模拟生成带有噪声的雷达信号
def generate_radar_signal(size):
    signal = np.zeros((size, size))
    # 在信号中添加模拟的雷达反射信号
    signal[size//2, size//2] = 10
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
            window = signal[i-pad:i+pad+1, j-pad:j+pad+1]
            window_median = np.median(window)
            window_range = np.max(window) - np.min(window)
            signal_value = signal[i, j]
            if signal_value > window_median - window_range and signal_value < window_median + window_range:
                filtered_signal[i, j] = signal_value
            else:
                filtered_signal[i, j] = window_median
    return filtered_signal

# 计算差异直方图
def compute_difference_histogram(original, processed):
    difference = np.abs(original - processed)
    return np.histogram(difference, bins=50, range=(0, np.max(difference)))

# 生成模拟的雷达信号
size = 100
radar_signal = generate_radar_signal(size)

# 对信号进行自适应中值滤波处理
window_size = 5  # 调整自适应中值滤波的窗口大小，以观察效果
processed_radar = adaptive_median_filter(radar_signal, window_size)

# 计算差异直方图
hist, bins = compute_difference_histogram(radar_signal, processed_radar)

# 显示信号和处理前后对比
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(radar_signal, cmap='viridis', vmin=0, vmax=10)
plt.title('Original Signal')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(processed_radar, cmap='viridis', vmin=0, vmax=10)
plt.title('Processed Signal')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(radar_signal - processed_radar, cmap='coolwarm')
plt.title('Difference')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.bar(bins[:-1], hist, width=bins[1]-bins[0], align='edge')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.title('Difference Histogram')

plt.tight_layout()
plt.show()

"""
代码在底部添加了一个差异直方图，用来更清楚地显示信号处理前后的差异。
直方图展示了差异的分布情况，可以帮助您更直观地理解滤波处理的效果。
通过观察直方图，您可以更好地判断滤波的影响，以及信号处理前后的变化。
"""