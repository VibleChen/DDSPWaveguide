import math

import torch

from Constant import SR, SpeedOfSound
from Losses import coeffs_norm


def fac(d, n, k):
    return (d - k) / ((n - k) + (n == k))


def select2(s, x, y):
    if s >= 1 or s <= -1:
        return y
    else:
        return x


# facs1(N,d,n) = select2(n,1,prod(k,max(1,n),select2(k<n,1,fac(d,n,k))));

def facs1(N, d, n):
    if -1 < n < 1:
        return 1
    else:
        prod = 1
        for k in range(max(1, n)):
            prod *= select2(k < n, 1, fac(d, n, k))
        return prod


# facs2(N,d,n) = select2(n<N,1,prod(l,max(1,N-n),fac(d,n,l+n+1)));
def facs2(N, d, n):
    if n >= N:
        return 1
    else:
        prod = 1
        for l in range(max(1, N - n)):
            prod *= fac(d, n, l + n + 1)
        return prod


# h(N,d,n) = facs1(N,d,n) * facs2(N,d,n);
def h(N, d, n):
    return facs1(N, d, n) * facs2(N, d, n)


# fdelayltv(N,n,d,x) = sum(i, N+1, delay(n,id+i,x) * h(N,fd,i))


def frac(x):
    return x - math.floor(x)


def l2s(l):
    return l * SR / SpeedOfSound


def s2l(s):
    return s * SpeedOfSound / SR


def f2l(f):
    return SpeedOfSound / f


def get_complex(reals, imgs):
    complex = torch.complex(reals, imgs)
    complex_conj = torch.conj(complex)

    return torch.cat((complex, complex_conj))


def get_coeffs_from_roots(roots):
    n = roots.size(0)
    coeffs = torch.zeros(n + 1, dtype=roots.dtype, device=roots.device)
    coeffs[0] = 1.0  # 最高次项系数设为 1
    for root in roots:
        # 创建新的多项式系数数组
        new_coeffs = torch.zeros_like(coeffs)
        # 更新多项式系数
        for i in range(n, 0, -1):
            new_coeffs[i] += coeffs[i - 1]
            new_coeffs[i - 1] -= root * coeffs[i - 1]
        coeffs = new_coeffs
    return torch.flip(coeffs.real, [0])


def get_freq_response(b, a, k=512):
    w = torch.linspace(0, torch.pi, k)
    jw = 1j * w
    num = sum(b[k] * torch.exp(-jw * k) for k in range(len(b)))
    den = sum(a[k] * torch.exp(-jw * k) for k in range(len(a)))
    H = num / den

    return H


def get_reals_and_imgs(params):
    reals = params[0]
    imgs = params[1]

    return reals, imgs


def get_trained_params(a_real, a_img, b_real, b_img):
    poles_complex = get_complex(a_real, a_img)
    a_coeffs = get_coeffs_from_roots(poles_complex)

    zeros_complex = get_complex(b_real, b_img)
    b_coeffs = get_coeffs_from_roots(zeros_complex)

    b_coeffs = coeffs_norm(b_coeffs, a_coeffs)

    filter_params = torch.stack([b_coeffs, a_coeffs])

    return filter_params


def get_filter_params(latent):
    nut_params = latent[:, 0]
    bridge_params = latent[:, 1]
    dispersion_params = latent[:, 2]

    return nut_params, bridge_params, dispersion_params
