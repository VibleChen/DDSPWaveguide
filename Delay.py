import math

import torch

"""
// Fourth-order case - delay d should be at least 1.5

fdelay4(n,d,x) =
    delay(n,id,x) * fdm1*fdm2*fdm3*fdm4/24
  + delay(n,id+1,x) * (0-fd*fdm2*fdm3*fdm4)/6
  + delay(n,id+2,x) * fd*fdm1*fdm3*fdm4/4
  + delay(n,id+3,x) * (0-fd*fdm1*fdm2*fdm4)/6
  + delay(n,id+4,x) * fd*fdm1*fdm2*fdm3/24
with {
  o = 1.49999;
  dmo = d - o; // assumed nonnegative
  id = int(dmo);
  fd = o + frac(dmo);
  fdm1 = fd-1;
  fdm2 = fd-2;
  fdm3 = fd-3;
  fdm4 = fd-4;
};"""


def fac(d, n, k):
    """

    :param d: [batch_size,]
    :param n:int
    :param k: [batch_size,]
    :return:
    """
    n = torch.tensor(n, dtype=torch.int)

    return (d - k) / (n - k + (n == k))


def select2(s, x, y):
    if s >= 1 or s <= -1:
        return y
    else:
        return x


def select2_batched(s, x, y):
    """
    Selects between x and y based on the condition s
    :param s:
    :param x:
    :param y:
    :return:
    """
    s = torch.tensor(s, dtype=torch.bool)
    return torch.where((s == 1), y, x)


def frac(x):
    return x - math.floor(x)


def facs1(N, d, n):
    if -1 < n < 1:
        return 1
    else:
        prod = 1
        for k in range(max(1, n)):
            prod *= select2(k < n, 1, fac(d, n, k))

        return prod


def facs1_batched(N, d, n):
    """

    :param N: int
    :param d: [batch_size,]
    :param n: int
    :return:
    """
    if n == 0:
        return torch.ones_like(d)

    prod = torch.ones_like(d)

    for k in range(max(1, n)):
        prod *= select2_batched(k < n, torch.ones_like(d), fac(d, n, k))

    return prod


def facs2(N, d, n):
    if n >= N:
        return 1
    else:
        prod = 1
        for l in range(max(1, N - n)):
            prod *= fac(d, n, l + n + 1)

        return prod


def facs2_batched(N, d, n):
    """

    :param N:
    :param d:
    :param n:
    :return:
    """
    if n >= N:
        return torch.ones_like(d)
    else:
        prod = torch.ones_like(d)
        for l in range(max(1, N - n)):
            prod *= fac(d, n, l + n + 1)

        return prod


def h(N, d, n):
    """

    :param N: int
    :param d: [batch_size,], in fraction
    :param n: int
    :return:
    """
    # facs1.shape = [B,] * facs2.shape = [B,]
    return facs1(N, d, n) * facs2(N, d, n)


def h_batched(N, d, n):
    """

    :param N: int
    :param d: [batch_size,], in fraction
    :param n: int
    :return:
    """
    # facs1.shape = [B,] * facs2.shape = [B,]
    return facs1_batched(N, d, n) * facs2_batched(N, d, n)


def delay(d: torch.tensor, x: torch.tensor):
    """
    Delay the signal by d samples
    :param d:
    :param x:
    :return:
    """
    if d == 0:
        return x
    zeros = torch.zeros(d, dtype=torch.float32)
    delayed_tensor = torch.cat((zeros, x[:-d]))
    return delayed_tensor


def delay_batched(d: torch.tensor, x: torch.tensor):
    batch_size, sequence_length = x.shape

    # Create a tensor of zeros to fill in for delays
    zeros = torch.zeros(batch_size, sequence_length, dtype=x.dtype, device=x.device)

    # Create indices for each element in the batch
    indices = torch.arange(sequence_length, dtype=torch.long, device=x.device) - d[:, None]

    # Create a mask to identify valid indices
    mask = indices >= 0

    # Adjust indices to be within valid range
    indices = torch.clamp(indices, min=0)

    # Use advanced indexing to gather the delayed elements and apply mask
    gathered_x = x.gather(1, indices)
    delayed_tensor = torch.where(mask, gathered_x, zeros)

    return delayed_tensor


def fdelay4(d, x):
    """
    Fractional delay with order of 4
    :param d: the delayline, shape is [B,]
    :param x: the signal, shape is [B, length]
    :return: delayed signal, shape is [B, length]
    """

    # o = 1.49999
    # dmo = d - o
    # assert dmo >= 0, "assumed nonnegative [d > 1.49999]"
    # id = int(dmo)
    # fd = o + frac(dmo)

    o = (N - 1.00001) / 2
    dmo = d - o
    assert (dmo > 0).all(), "assumed nonnegative [d > (N-1)/2]"

    id = dmo.floor().to(torch.int)

    fd = o + (dmo - dmo.floor())

    fdm1 = fd - 1
    fdm2 = fd - 2
    fdm3 = fd - 3
    fdm4 = fd - 4

    delays = [
        delay_batched(id, x) * (fdm1 * fdm2 * fdm3 * fdm4 / 24).view(-1, 1),
        delay_batched(id + 1, x) * ((-fd * fdm2 * fdm3 * fdm4) / 6).view(-1, 1),
        delay_batched(id + 2, x) * (fd * fdm1 * fdm3 * fdm4 / 4).view(-1, 1),
        delay_batched(id + 3, x) * ((-fd * fdm1 * fdm2 * fdm4) / 6).view(-1, 1),
        delay_batched(id + 4, x) * (fd * fdm1 * fdm2 * fdm3 / 24).view(-1, 1),
    ]

    return sum(delays)


def fdelayltv(N, d, x):
    o = (N - 1.00001) / 2
    dmo = d - o
    assert dmo > 0, "assumed nonnegative [d > (N-1)/2]"
    id = int(dmo)
    fd = o + frac(dmo)
    sum = 0
    for i in range(N + 1):
        sum += delay(id + i, x) * h(N, fd, i)

    return sum


def fdelayltv_batched(N, d, x):
    """

    :param N: the order of lagrange interpolation, default is 4.
    :param d: the delayline, shape is [B,]
    :param x: the signal, shape is [B, length]
    :return:
    """
    o = (N - 1.00001) / 2
    dmo = d - o
    assert (dmo > 0).all(), "assumed nonnegative [d > (N-1)/2]"
    id = dmo.floor().to(torch.int)
    fd = o + (dmo - dmo.floor())

    delayed_sum = torch.zeros_like(x)

    for i in range(N + 1):
        delayed_sum += delay_batched(id + i, x) * h(N, fd, i).view(-1, 1)

    return delayed_sum


if __name__ == "__main__":
    batch_size = 3
    sequence_length = 32000
    # N = torch.tensor(4)
    d = torch.tensor([1, 1, 1], dtype=torch.float32)
    x = torch.randn(batch_size, sequence_length, requires_grad=True)
    print(delay_batched(d, x))
