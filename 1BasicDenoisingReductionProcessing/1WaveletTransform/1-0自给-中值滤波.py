import numpy as np
import cv2
import matplotlib.pyplot as plt

# 读取原始图像
image_path = 'path_to_your_image.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 添加椒盐噪声
def add_salt_and_pepper_noise(image, amount=0.04):
    noisy_image = np.copy(image)
    num_salt = int(amount * image.size)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    num_pepper = int(amount * image.size)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

noisy_image = add_salt_and_pepper_noise(original_image)

# 中值滤波
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

filtered_image = median_filter(noisy_image, kernel_size=3)

# 绘制对比图像
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Median)')

plt.tight_layout()
plt.show()
