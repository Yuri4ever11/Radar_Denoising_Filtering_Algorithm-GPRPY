import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_dzt_file(filename):
    # 读取DZT文件并返回数据和头部信息
    info = {}
    fid = open(filename, 'rb')

    # Read the header information
    info["rh_tag"] = struct.unpack('h', fid.read(2))[0]
    info["rh_data"] = struct.unpack('h', fid.read(2))[0]
    info["rh_nsamp"] = struct.unpack('h', fid.read(2))[0]
    info["rh_bits"] = struct.unpack('h', fid.read(2))[0]
    info["rhf_sps"] = struct.unpack('f', fid.read(4))[0]
    info["rhf_spm"] = struct.unpack('f', fid.read(4))[0]
    info["rhf_position"] = struct.unpack('f', fid.read(4))[0]
    info["rhf_range"] = struct.unpack('f', fid.read(4))[0]
    info["rh_npass"] = struct.unpack('h', fid.read(2))[0]
    info["rhb_cdt"] = struct.unpack('f', fid.read(4))[0]
    info["rhb_mdt"] = struct.unpack('f', fid.read(4))[0]
    info["rh_mapOffset"] = struct.unpack('h', fid.read(2))[0]
    info["rh_mapSize"] = struct.unpack('h', fid.read(2))[0]
    info["rh_text"] = struct.unpack('h', fid.read(2))[0]
    info["rh_ntext"] = struct.unpack('h', fid.read(2))[0]
    info["rh_proc"] = struct.unpack('h', fid.read(2))[0]
    info["rh_nproc"] = struct.unpack('h', fid.read(2))[0]
    info["rh_nchan"] = struct.unpack('h', fid.read(2))[0]

    if info["rh_data"] < 1024:
        offset = 1024 * info["rh_data"]
    else:
        offset = 1024 * info["rh_nchan"]

    if info["rh_bits"] == 8:
        datatype = 'uint8'
    elif info["rh_bits"] == 16:
        datatype = 'uint16'
    elif info["rh_bits"] == 32:
        datatype = 'int32'

    # Read the entire file
    vec = np.fromfile(filename, dtype=datatype)
    headlength = offset / (info["rh_bits"] / 8)
    datvec = vec[int(headlength):]

    if info["rh_bits"] == 8 or info["rh_bits"] == 16:
        datvec = datvec - (2 ** info["rh_bits"]) / 2.0

    data = np.reshape(datvec, (info["rh_nsamp"], -1), order='F')  # Reshape data correctly

    fid.close()

    return data, info
# 自定义处理函数示例：对比度和亮度调整
def adjust_contrast_brightness(data, contrast_factor, brightness_factor):
    adjusted_data = (data - 128) * contrast_factor + 128 + brightness_factor
    adjusted_data = np.clip(adjusted_data, 0, 255)

    return adjusted_data


def remMeanTrace(data, ntraces):
    '''
    Subtracts from each trace the average trace over
    a moving average window.

    Can be used to remove horizontal arrivals,
    such as the airwave.

    INPUT:
    data       data matrix whose columns contain the traces
    ntraces    window width; over how many traces
               to take the moving average.

    OUTPUT:
    newdata    data matrix after subtracting average traces
    '''

    data = np.asmatrix(data)
    tottraces = data.shape[1]
    # For ridiculous ntraces values, just remove the entire average
    if ntraces >= tottraces:
        newdata = data - np.matrix.mean(data, 1)
    else:
        newdata = np.asmatrix(np.zeros(data.shape))
        halfwid = int(np.ceil(ntraces / 2.0))

        # First few traces, that all have the same average
        avgtr = np.matrix.mean(data[:, 0:halfwid + 1], 1)
        newdata[:, 0:halfwid + 1] = data[:, 0:halfwid + 1] - avgtr

        # For each trace in the middle
        for tr in tqdm(range(halfwid, tottraces - halfwid + 1)):
            winstart = int(tr - halfwid)
            winend = int(tr + halfwid)
            avgtr = np.matrix.mean(data[:, winstart:winend + 1], 1)
            newdata[:, tr] = data[:, tr] - avgtr

        # Last few traces again have the same average
        avgtr = np.matrix.mean(data[:, tottraces - halfwid:tottraces + 1], 1)
        newdata[:, tottraces - halfwid:tottraces + 1] = data[:, tottraces - halfwid:tottraces + 1] - avgtr

    print('done with removing mean trace')
    return newdata

# 保存数据到Excel文件
def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False, header=False)

# 保存数据为普通图像
def save_as_image(data, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='gray', aspect='auto')
    plt.colorbar()
    plt.title("Processed Data")
    plt.xlabel("Samples")
    plt.ylabel("Traces")
    plt.savefig(filename)
    plt.close()

# 保存数据为普通图像
def save_as_normal_image(data, filename):
    plt.figure(figsize=(10, 6))
    plt.imshow(data, cmap='viridis', aspect='auto')  # 使用不同的colormap
    plt.colorbar()
    plt.title("Processed Data")
    plt.xlabel("Samples")
    plt.ylabel("Traces")
    plt.savefig(filename)
    plt.close()

# 保存数据为DZT文件
def save_as_dzt(data, info, filename):
    # 创建一个DZT文件
    with open(filename, 'wb') as f:
        # 写入头部信息
        header = struct.pack('hhhhff',
                             info["rh_tag"], info["rh_data"], info["rh_nsamp"], info["rh_bits"], info["rhf_sps"],
                             info["rhf_spm"])
        f.write(header)

        # 写入数据部分
        data_bytes = data.tobytes()
        f.write(data_bytes)

# 主处理函数
def main():
    # 输入DZT文件路径，请替换为您的DZT文件路径
    dzt_filename = 'H:\D_ProjectFiles\P1_Project145\Step3SoftWare\process\dewow/dewowed.dzt'  # 替换为您的DZT文件路径

    # 读取DZT文件和头部信息
    data, info = read_dzt_file(dzt_filename)

    # 自定义对比度和亮度调整因子，根据需要进行调整
    contrast_factor = 0.0001  # 对比度增强因子
    brightness_factor = 0.0001  # 亮度增强因子

    # 调整对比度和亮度
    adjusted_data = adjust_contrast_brightness(data, contrast_factor, brightness_factor)

    # 自定义移除平均轨迹的窗口大小，根据需要设置窗的大小
    mean_trace_window = 1111

    # 移除平均轨迹
    removed_mean_trace_data = remMeanTrace(adjusted_data, mean_trace_window)

    # 保存处理后的数据到Excel文件
    save_to_excel(removed_mean_trace_data, 'removed_mean_trace_data.xlsx')

    # 保存处理后的数据为普通图像
    save_as_image(removed_mean_trace_data, 'removed_mean_trace_image.png')

    # 保存处理后的数据为普通图像（使用不同的colormap）
    save_as_normal_image(removed_mean_trace_data, 'removed_mean_trace_normal_image.png')

    # 保存处理后的数据为DZT文件
    save_as_dzt(removed_mean_trace_data, info, 'removed_mean_trace_data.dzt')


if __name__ == "__main__":
    main()
