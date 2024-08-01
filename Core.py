import math

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
