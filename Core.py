import math

import torch

from Constant import SR, SpeedOfSound


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

    return torch.cat((complex, complex_conj), dim=-1)


def get_coeffs_from_roots(roots):
    bs, n = roots.shape
    coeffs = torch.zeros((bs, n + 1), dtype=roots.dtype, device=roots.device)
    coeffs[:, 0] = 1.0  # 最高次项系数设为 1
    for i in range(n):
        root = roots[:, i]
        # 创建新的多项式系数数组
        new_coeffs = torch.zeros_like(coeffs)
        # 更新多项式系数
        for j in range(n, 0, -1):
            new_coeffs[:, j] += coeffs[:, j - 1]
            new_coeffs[:, j - 1] -= root * coeffs[:, j - 1]
        coeffs = new_coeffs
    return torch.flip(coeffs.real, dims=[1])


def get_freq_response(b, a, k=512):
    bs, n_b = b.shape
    _, n_a = a.shape

    w = torch.linspace(0, torch.pi, k).unsqueeze(0).expand(bs, -1)
    jw = 1j * w

    num = sum(b[:, k].unsqueeze(1) * torch.exp(-jw * k) for k in range(n_b))
    den = sum(a[:, k].unsqueeze(1) * torch.exp(-jw * k) for k in range(n_a))

    H = num / den

    return H


def get_reals_and_imgs(params):
    batch_size = params.size(0)
    params = params.reshape(batch_size, 4, -1)
    a_reals = params[:, 0]
    a_imgs = params[:, 1]
    b_reals = params[:, 2]
    b_imgs = params[:, 3]

    return a_reals, a_imgs, b_reals, b_imgs


def get_trained_params(a_real, a_img, b_real, b_img):
    poles_complex = get_complex(a_real, a_img)
    a_coeffs = get_coeffs_from_roots(poles_complex)

    zeros_complex = get_complex(b_real, b_img)
    b_coeffs = get_coeffs_from_roots(zeros_complex)

    b_coeffs = coeffs_norm(b_coeffs, a_coeffs)

    filter_params = torch.stack([b_coeffs, a_coeffs])

    return filter_params


def get_filter_params(filter_params):
    batch_size = filter_params.size(0)
    filter_params = filter_params.view(batch_size, 3, -1)
    nut_params = filter_params[:, 0]
    bridge_params = filter_params[:, 1]
    dispersion_params = filter_params[:, 2]

    return nut_params, bridge_params, dispersion_params


def coeffs_norm(b_coeffs, a_coeffs):
    fr = get_freq_response(b_coeffs, a_coeffs)
    ar = torch.abs(fr)

    max_ar = ar.max(dim=1, keepdim=True)[0]
    norm_b_coeffs = b_coeffs / max_ar

    return norm_b_coeffs


def get_filters_coeffs(filter_params):
    nut_params, bridge_params, dispersion_params = get_filter_params(filter_params)
    nut_a_reals, nut_a_imgs, nut_b_reals, nut_b_imgs = get_reals_and_imgs(nut_params)
    bridge_a_reals, bridge_a_imgs, bridge_b_reals, bridge_b_imgs = get_reals_and_imgs(bridge_params)
    dispersion_a_reals, dispersion_a_imgs, dispersion_b_reals, dispersion_b_imgs = get_reals_and_imgs(
        dispersion_params)

    nut_poles_complex = get_complex(nut_a_reals, nut_a_imgs)
    nut_zeros_complex = get_complex(nut_b_reals, nut_b_imgs)
    nut_a_coeffs = get_coeffs_from_roots(nut_poles_complex)
    nut_b_coeffs = coeffs_norm(get_coeffs_from_roots(nut_zeros_complex), nut_a_coeffs)

    # print(f"nut_a_coeffs: {nut_a_coeffs}")
    # print(f"nut_b_coeffs: {nut_b_coeffs}")

    bridge_poles_complex = get_complex(bridge_a_reals, bridge_a_imgs)
    bridge_zeros_complex = get_complex(bridge_b_reals, bridge_b_imgs)
    bridge_a_coeffs = get_coeffs_from_roots(bridge_poles_complex)
    bridge_b_coeffs = coeffs_norm(get_coeffs_from_roots(bridge_zeros_complex), bridge_a_coeffs)

    # print(f"bridge_a_coeffs: {bridge_a_coeffs}")
    # print(f"bridge_b_coeffs: {bridge_b_coeffs}")

    dispersion_poles_complex = get_complex(dispersion_a_reals, dispersion_a_imgs)
    dispersion_zeros_complex = get_complex(dispersion_b_reals, dispersion_b_imgs)
    dispersion_a_coeffs = get_coeffs_from_roots(dispersion_poles_complex)
    dispersion_b_coeffs = coeffs_norm(get_coeffs_from_roots(dispersion_zeros_complex), dispersion_a_coeffs)

    # print(f"dispersion_a_coeffs: {dispersion_a_coeffs}")
    # print(f"dispersion_b_coeffs: {dispersion_b_coeffs}")

    return nut_a_coeffs, nut_b_coeffs, bridge_a_coeffs, bridge_b_coeffs, dispersion_a_coeffs, dispersion_b_coeffs
