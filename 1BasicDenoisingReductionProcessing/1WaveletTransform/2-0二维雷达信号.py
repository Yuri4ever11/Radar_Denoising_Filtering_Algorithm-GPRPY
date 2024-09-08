import numpy as np
import matplotlib.pyplot as plt

# 设置图像大小
width, height = 512, 512

# 生成均匀分布的随机数据（模拟雷达信号）
radar_signal = np.random.uniform(0, 1, size=(width, height))

# 添加高斯噪声
noise = np.random.normal(0, 0.1, size=(width, height))
noisy_radar_signal = radar_signal + noise

# 绘制原始雷达信号图像
plt.subplot(1, 2, 1)
plt.imshow(radar_signal, cmap='gray')
plt.title('Original Radar Signal')
plt.axis('off')

# 绘制带噪声的雷达信号图像
plt.subplot(1, 2, 2)
plt.imshow(noisy_radar_signal, cmap='gray')
plt.title('Noisy Radar Signal')
plt.axis('off')

plt.show()
