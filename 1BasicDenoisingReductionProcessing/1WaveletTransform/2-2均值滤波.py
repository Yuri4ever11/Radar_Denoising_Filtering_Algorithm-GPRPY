import numpy as np
import matplotlib.pyplot as plt
import cv2

# 设置图像大小
width, height = 256, 256

# 生成均匀分布的随机数据（模拟雷达信号）
radar_signal = np.random.uniform(0, 1, size=(width, height))

# 添加高斯噪声
noise = np.random.normal(0, 0.1, size=(width, height))
noisy_radar_signal = radar_signal + noise

# 应用均值滤波
kernel_size = 15  # 滤波器大小
filtered_radar_signal = cv2.blur(noisy_radar_signal, (kernel_size, kernel_size))

# 绘制原始雷达信号、带噪声的雷达信号和经过均值滤波后的雷达信号图像
plt.subplot(1, 3, 1)
plt.imshow(radar_signal, cmap='gray')
plt.title('Original Radar Signal')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_radar_signal, cmap='gray')
plt.title('Noisy Radar Signal')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(filtered_radar_signal, cmap='gray')
plt.title('Filtered Radar Signal')
plt.axis('off')

plt.tight_layout()
plt.show()
