import numpy as np
import cv2
import matplotlib.pyplot as plt

# 生成复杂的二维雷达图像
size = 256
x = np.linspace(-10, 10, size)
y = np.linspace(-10, 10, size)
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)
radar_image = np.sin(10*r) * np.cos(5*theta)



# 显示原始图像
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# 添加噪声
noise = np.random.normal(0, 0.2, (size, size))
noisy_radar_image = radar_image + noise

# 进行均值滤波处理
kernel_size = 5
filtered_image = cv2.blur(noisy_radar_image, (kernel_size, kernel_size))

# 显示处理前后的对比图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(radar_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(noisy_radar_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()
"""
首先生成了一个复杂的二维雷达图像，然后添加了高斯噪声。
接着使用OpenCV库的均值滤波函数 cv2.blur 对噪声图像进行均值滤波处理。
最后，我们使用Matplotlib库显示了处理前后的对比图像，分别展示了原始图像、带噪声图像以及均值滤波后的图像。
你可以根据需要调整参数来生成不同的图像并观察均值滤波的效果。
"""