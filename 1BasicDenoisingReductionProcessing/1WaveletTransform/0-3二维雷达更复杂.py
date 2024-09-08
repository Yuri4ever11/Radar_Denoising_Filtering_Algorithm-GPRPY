import numpy as np
import matplotlib.pyplot as plt
import pywt

# 生成原始二维图像
image_size = 256
x = np.linspace(0, 4 * np.pi, image_size)
y = np.linspace(0, 4 * np.pi, image_size)
x, y = np.meshgrid(x, y)
original_image = np.sin(x) + np.cos(y) + np.random.normal(0, 0.5, (image_size, image_size))

# 可调参数及其作用：
wavelet = 'db4'  # 小波基函数
"""
小波基函数（Wavelet）用于进行小波分解和重构，不同的小波基函数适用于不同类型的信号和图像。以下是一些常见的小波基函数：

'haar'：Haar小波，适用于处理分段常数信号。
'dbN'：Daubechies小波，N为小波的阶数，例如'db2'、'db4'等，适用于平滑或较平稳的信号。
'symN'：Symlets小波，N为小波的阶数，类似于Daubechies小波但对称性更好。
'coifN'：Coiflets小波，N为小波的阶数，适用于图像边缘检测和分割。
'biorN.M'：Biorthogonal小波，N和M分别为两个滤波器的阶数，适用于某些信号的特定应用。
您可以根据信号的特点和需求选择适合的小波基函数。
"""
level = 3  # 小波分解层数
threshold_type = 'soft'  # 阈值类型，'soft'或'hard'
"""
在小波去噪中，阈值类型用于决定保留还是丢弃小波系数，从而实现去噪。有两种常见的阈值类型：

'soft'：软阈值，将小于阈值的系数设为0，大于阈值的系数减去阈值。
'hard'：硬阈值，将小于阈值的系数设为0，保留大于阈值的系数。
选择软阈值或硬阈值取决于信号噪声的性质和预期的去噪效果。

在实际应用中，您可以尝试不同的小波基函数和阈值类型组合，通过观察去噪效果来选择最合适的参数。
"""
threshold_value = 0.5  # 阈值，用于控制去噪强度

# 进行小波去噪
coeffs = pywt.wavedec2(original_image, wavelet, level=level)
coeffs = list(coeffs)
for i in range(1, level + 1):
    coeffs[i] = [pywt.threshold(j, value=threshold_value, mode=threshold_type) for j in coeffs[i]]
denoised_image = pywt.waverec2(coeffs, wavelet)

# 绘制原始图像和去噪后的图像对比图
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray', extent=[0, 4 * np.pi, 0, 4 * np.pi], origin='upper')
plt.title('Original Image')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(denoised_image, cmap='gray', extent=[0, 4 * np.pi, 0, 4 * np.pi], origin='upper')
plt.title('Denoised Image')
plt.colorbar()

plt.tight_layout()
plt.show()
"""
小波变换在降噪中的核心思想是将信号分解成不同尺度的小波系数，
然后根据阈值进行系数的调整，最终重构出去除噪声的信号。

在小波分解过程中，高频部分通常会包含噪声，因为噪声在高频范围内更为显著。
通过适当的阈值处理，小波变换可以将噪声对应的高频小波系数设为零，从而实现噪声的去除。
低频部分通常包含信号的主要信息，因此保留这些部分有助于保持信号的主要特征。

通过阈值处理，小波变换实际上是在保留信号主要特征的同时去除了噪声部分，
因此在视觉上可以观察到图像或信号变得更加清晰。这是小波变换在信号降噪中的优势之一。
"""