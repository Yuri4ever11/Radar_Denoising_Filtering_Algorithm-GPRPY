import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
#
# # 生成一个带有噪声的二维图像
# image_size = 256
# image = np.random.random((image_size, image_size))
#
# # 添加噪声
# noise_level = 0.2
# image_with_noise = image + noise_level * np.random.random((image_size, image_size))
#
# # 应用中值滤波进行降噪
# filtered_image = medfilt(image_with_noise)
#
# # 绘制对比图像
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')
#
# plt.subplot(1, 3, 2)
# plt.imshow(image_with_noise, cmap='gray')
# plt.title('Image with Noise')
# plt.axis('off')
#
# plt.subplot(1, 3, 3)
# plt.imshow(filtered_image, cmap='gray')
# plt.title('Filtered Image')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 生成一个复杂的二维雷达图像
np.random.seed(0)
image = np.random.random((300, 300)) + np.sin(np.linspace(0, 3 * np.pi, 300))
image = np.clip(image, 0, 1)

# 显示原始图像
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# 中值滤波处理
filtered_image = signal.medfilt2d(image, kernel_size=3)  # 调整kernel_size参数
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Median Filtered Image')

plt.tight_layout()
plt.show()
