import numpy as np
import matplotlib.pyplot as plt


# 生成含有干扰的信号（示例）
def generate_signal(size, freq_interference, amplitude_interference):
    x = np.linspace(0, 1, size)
    signal = np.sin(2 * np.pi * x * 5) + amplitude_interference * np.sin(2 * np.pi * x * freq_interference)
    return signal


# 自适应陷波滤波器
def adaptive_notch_filter(signal, interference_freq, step_size, order):
    filtered_signal = np.zeros_like(signal)
    weights = np.zeros(order + 1)

    for i in range(order, len(signal)):
        input_vector = np.flip(signal[i - order:i + 1])  # 输入向量，包括过去的信号值
        output = np.dot(weights, input_vector)  # 滤波器输出

        # 更新权重（使用LMS算法）
        error = signal[i] - output
        weights = weights + step_size * error * input_vector

        filtered_signal[i] = output

    return filtered_signal


# 生成含有干扰的信号
size = 1000
interference_freq = 50  # 干扰频率
amplitude_interference = 0.5  # 干扰幅度
signal = generate_signal(size, interference_freq, amplitude_interference)

# 自适应陷波滤波参数
step_size = 0.01  # 步长，用于权重更新
order = 10  # 滤波器阶数

# 对信号进行自适应陷波滤波处理
filtered_signal = adaptive_notch_filter(signal, interference_freq, step_size, order)

# 显示信号和处理前后对比
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Original Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(filtered_signal)
plt.title('Filtered Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
