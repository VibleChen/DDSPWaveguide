import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import signal

from Core import get_complex, get_coeffs_from_roots,coeffs_norm

device = 'cpu'


def allpass_filter_coeff_complex(n, method='butter', fc=0.25, ripple=None):
    """
    生成n阶全通滤波器的零点和极点，并将它们分解为实部和虚部

    参数:
    n : int
        滤波器的阶数
    method : str, 可选
        滤波器设计方法 ('butter', 'cheby1', 'bessel')
    fc : float, 可选
        截止频率 (归一化, 0 < fc < 0.5)
    ripple : float, 可选
        切比雪夫滤波器的纹波 (dB)

    返回:
    b_real : ndarray
        分子系数的实部
    b_imag : ndarray
        分子系数的虚部
    a_real : ndarray
        分母系数的实部
    a_imag : ndarray
        分母系数的虚部
    """
    if method not in ['butter', 'cheby1', 'bessel']:
        raise ValueError("Unsupported filter method")

    # 生成低通滤波器的零点和极点
    if method == 'butter':
        z, p, k = signal.butter(n, fc, btype='low', output='zpk')
    elif method == 'cheby1':
        if ripple is None:
            ripple = 1.0  # 默认纹波
        z, p, k = signal.cheby1(n, ripple, fc, btype='low', output='zpk')
    elif method == 'bessel':
        z, p, k = signal.bessel(n, fc, btype='low', output='zpk', norm='phase')

    # 将低通滤波器转换为全通滤波器
    ap_z = 1 / np.conj(p)  # 全通滤波器的零点
    ap_p = p  # 全通滤波器的极点

    # 确保增益为1
    ap_k = 1.0

    # 转换为传递函数系数
    b, a = signal.zpk2tf(ap_z, ap_p, ap_k)

    # 计算频率响应用于正则化
    w, h = signal.freqz(b, a)
    b = b / np.max(np.abs(h))

    z, p, _ = signal.tf2zpk(b, a)
    print(b)
    print(a)

    # 分解传递函数系数为实部和虚部
    z_real = np.real(z)
    z_imag = np.imag(z)
    p_real = np.real(p)
    p_imag = np.imag(p)

    return z_real[1::2], np.abs(z_imag[1::2]), p_real[1::2], np.abs(p_imag[1::2])


def plot_response_complex(a_real, a_imag, b_real, b_imag, title):
    b = b_real + 1j * b_imag
    a = a_real + 1j * a_imag
    w, h = signal.freqz(b, a)

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(w / np.pi, np.abs(h))
    plt.title('Amplitude Response')
    plt.ylabel('Amplitude')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(w / np.pi, np.unwrap(np.angle(h)))
    plt.title('Phrase Response')
    plt.ylabel('Phrase (Rad)')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    nut_z_real, nut_z_imag, nut_p_real, nut_p_imag = allpass_filter_coeff_complex(4)
    # print (nut_z_real, nut_z_imag, nut_p_real, nut_p_imag)

    nut_z_real = torch.tensor(nut_z_real, dtype=torch.float32, requires_grad=True)
    nut_z_imag = torch.tensor(nut_z_imag, dtype=torch.float32, requires_grad=True)
    nut_p_real = torch.tensor(nut_p_real, dtype=torch.float32, requires_grad=True)
    nut_p_imag = torch.tensor(nut_p_imag, dtype=torch.float32, requires_grad=True)

    z_compx = get_complex(nut_z_real.unsqueeze(0), nut_z_imag.unsqueeze(0))
    p_compx = get_complex(nut_p_real.unsqueeze(0), nut_p_imag.unsqueeze(0))

    nut_a_coeff = get_coeffs_from_roots(p_compx)
    nut_b_coeff = coeffs_norm(get_coeffs_from_roots(z_compx),nut_a_coeff)

    print (nut_b_coeff,nut_a_coeff)
