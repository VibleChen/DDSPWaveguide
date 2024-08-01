import torch

from Core import h, frac

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


def delay(d, x):
    if d == 0:
        return x
    zeros = torch.zeros(d, dtype=torch.float32)
    delayed_tensor = torch.cat((zeros, x[:-d]))
    return delayed_tensor


def fdelay4(d, x):
    """
    Fractional delay
    """

    o = 1.49999
    dmo = d - o
    assert dmo >= 0, "assumed nonnegative [d > 1.49999]"
    id = int(dmo)
    fd = o + frac(dmo)
    fdm1 = fd - 1
    fdm2 = fd - 2
    fdm3 = fd - 3
    fdm4 = fd - 4

    delays = [
        delay(id, x) * fdm1 * fdm2 * fdm3 * fdm4 / 24,
        delay(id + 1, x) * (-fd * fdm2 * fdm3 * fdm4) / 6,
        delay(id + 2, x) * fd * fdm1 * fdm3 * fdm4 / 4,
        delay(id + 3, x) * (-fd * fdm1 * fdm2 * fdm4) / 6,
        delay(id + 4, x) * fd * fdm1 * fdm2 * fdm3 / 24
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
