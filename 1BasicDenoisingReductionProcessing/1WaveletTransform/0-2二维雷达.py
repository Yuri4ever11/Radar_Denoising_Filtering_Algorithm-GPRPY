import numpy as np
import matplotlib.pyplot as plt
import pywt

"""
首先生成了一个二维的原始雷达信号（使用正弦函数生成，并加入了噪声），然后使用小波变换进行去噪。
最后，使用Matplotlib绘制了原始信号和去噪后的信号对比图。
您可以根据需要调整小波基函数、分解层数以及阈值等参数来观察不同的效果。
"""
# 生成原始二维雷达信号
fs = 100  # 采样率
t = np.linspace(0, 1, fs, endpoint=False)  # 时间轴
x, y = np.meshgrid(t, t)
original_signal = np.sin(2 * np.pi * 5 * x) * np.sin(2 * np.pi * 10 * y) + np.random.normal(0, 0.5, (fs, fs))

# 进行小波去噪
wavelet = 'haar'  # 小波基函数
level = 3  # 小波分解层数
coeffs = pywt.wavedec2(original_signal, wavelet, level=level)
coeffs = list(coeffs)
for i in range(1, level + 1):
    coeffs[i] = [pywt.threshold(j, value=0.5, mode='soft') for j in coeffs[i]]
denoised_signal = pywt.waverec2(coeffs, wavelet)

# 绘制原始信号和去噪后的信号对比图
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_signal, cmap='gray', extent=[0, 1, 0, 1], origin='upper')
plt.title('Original Signal')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(denoised_signal, cmap='gray', extent=[0, 1, 0, 1], origin='upper')
plt.title('Denoised Signal')
plt.colorbar()

plt.tight_layout()
plt.show()
