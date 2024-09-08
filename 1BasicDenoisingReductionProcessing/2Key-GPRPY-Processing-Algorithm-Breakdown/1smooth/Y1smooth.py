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


def smooth(data, window):
    '''
    Replaces each sample along each trace with an
    along-time moving average.

    Can be used as high-cut filter.

    INPUT:
    data      data matrix whose columns contain the traces
    window    length of moving average window
              [in "number of samples"]

    OUTPUT:
    newdata   data matrix after applying smoothing
    '''
    totsamps = data.shape[0]
    # If the window is larger or equal to the number of samples,
    # then we can do a much faster dewow
    if (window >= totsamps):
        newdata = np.mean(data, axis=0)
    elif window == 1:
        newdata = data
    elif window == 0:
        newdata = data
    else:
        newdata = np.asmatrix(np.zeros(data.shape))
        halfwid = int(np.ceil(window / 2.0))

        # For the first few samples, it will always be the same
        newdata[0:halfwid + 1, :] = np.mean(data[0:halfwid + 1, :], axis=0)

        # for each sample in the middle
        for smp in tqdm(range(halfwid, totsamps - halfwid + 1)):
            winstart = int(smp - halfwid)
            winend = int(smp + halfwid)
            newdata[smp, :] = np.mean(data[winstart:winend + 1, :], axis=0)

        # For the last few samples, it will always be the same
        newdata[totsamps - halfwid:totsamps + 1, :] = np.mean(data[totsamps - halfwid:totsamps + 1, :], axis=0)

    print('done with smoothing')
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
    dzt_filename = 'H:\D_ProjectFiles\P1_Project145\Step3SoftWare\Test1\data/FILE____032.DZT'  # 替换为您的DZT文件路径

    # 读取DZT文件和头部信息
    data, info = read_dzt_file(dzt_filename)

    # 自定义对比度和亮度调整因子，根据需要进行调整
    contrast_factor = 0.001  # 对比度增强因子
    brightness_factor = 0.000001  # 亮度增强因子

    # 调整对比度和亮度
    adjusted_data = adjust_contrast_brightness(data, contrast_factor, brightness_factor)

    # 自定义平滑窗口大小
    smooth_window_size = 11  # 根据需要设置窗口大小

    # 应用平滑处理
    smoothed_data = smooth(adjusted_data, smooth_window_size)

    # 保存处理后的数据到Excel文件
    save_to_excel(smoothed_data, 'smoothed_data.xlsx')

    # 保存处理后的数据为普通图像
    save_as_image(smoothed_data, 'smoothed_image.png')

    # 保存处理后的数据为普通图像（使用不同的colormap）
    save_as_normal_image(smoothed_data, 'smoothed_normal_image.png')

    # 保存处理后的数据为DZT文件
    save_as_dzt(smoothed_data, info, 'smoothed_data.dzt')

if __name__ == "__main__":
    main()
