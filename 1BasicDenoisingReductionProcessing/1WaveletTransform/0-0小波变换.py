import numpy as np
import matplotlib.pyplot as plt
import pywt

# 生成带噪声的信号
t = np.linspace(0, 1, 500)
clean_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
noise = np.random.normal(0, 0.5, len(t))
noisy_signal = clean_signal + noise

# 进行小波分解和去噪处理
wavelet = 'db4'  # 选择小波函数
level = 4       # 尺度级别
coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
threshold = 0.3 * np.sqrt(2 * np.log(len(noisy_signal)))  # 阈值选择
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold)

# 重构信号
denoised_signal = pywt.waverec(coeffs, wavelet)

# 绘制原始信号、带噪声信号和去噪后的信号
plt.figure(figsize=(10, 6))
plt.plot(t, clean_signal, label='Clean Signal', color='blue')
plt.plot(t, noisy_signal, label='Noisy Signal', color='red', alpha=0.5)
plt.plot(t, denoised_signal, label='Denoised Signal', color='green')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Wavelet Denoising Example')
plt.show()
