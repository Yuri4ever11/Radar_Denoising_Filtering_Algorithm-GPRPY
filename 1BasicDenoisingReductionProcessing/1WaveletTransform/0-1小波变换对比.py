import numpy as np
import matplotlib.pyplot as plt
import pywt

# 生成原始信号
fs = 1000  # 采样率
t = np.linspace(0, 1, fs, endpoint=False)  # 时间轴
freq = 5  # 信号频率
original_signal = np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.5, fs)  # 加入噪声

# 进行小波去噪
wavelet = 'db4'  # 小波基函数
level = 5  # 小波分解层数
coeffs = pywt.wavedec(original_signal, wavelet, level=level)
coeffs[1:] = (pywt.threshold(i, value=0.5, mode='soft') for i in coeffs[1:])
denoised_signal = pywt.waverec(coeffs, wavelet)

# 绘制原始信号和去噪后的信号对比图
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, original_signal, label='Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Signal')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, denoised_signal, label='Denoised Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Denoised Signal')
plt.legend()

plt.tight_layout()
plt.show()
