import numpy as np
import torch
import torch.nn as nn
import torchaudio

from Constant import SR
from Delay import delay


def NonIdealFilter(signal):
    return signal * -0.999


def IdealFilter(signal):
    return signal * -1


class BridgeFilter(nn.Module):
    """
       bridgeFilter(brightness,absorption,x) = rho * (h0 * x' + h1*(x+x''))
   with{
       freq = 320;
       t60 = (1-absorption)*20;
       h0 = (1.0 + brightness)/2;
       h1 = (1.0 - brightness)/4;
       rho = pow(0.001,1.0/(freq*t60));
   };
   """

    def __init__(self, brightness, absorption, trainable=False):
        super().__init__()
        self.trainable = trainable
        if not trainable:
            freq = 320
            t60 = (1 - absorption) * 20
            self.h0 = (1.0 + brightness) / 2
            self.h1 = (1.0 - brightness) / 4
            self.rho = pow(0.001, 1.0 / (freq * t60))

    def forward(self, x, a_coeff=None, b_coeff=None):
        if not self.trainable:
            h1_part = self.h1 * (x + delay(2, x))
            h0_part = self.h0 * delay(1, x)
            return - self.rho * (h0_part + h1_part)
        else:
            assert a_coeff is not None and b_coeff is not None, "a_coeff and b_coeff should be provided when trainable is True"

            return - torchaudio.functional.lfilter(x, a_coeff, b_coeff)


class ModelFilter(nn.Module):
    """
    tf2(b0,b1,b2,a1,a2) = iir((b0,b1,b2),(1,a1,a2));

    modeFilter(freq,t60,gain) = fi.tf2(b0,b1,b2,a1,a2)*gain
with{
    b0 = 1;
    b1 = 0;
    b2 = -1;
    w = 2*ma.PI*freq/ma.SR;
    r = pow(0.001,1/float(t60*ma.SR));
    a1 = -2*r*cos(w);
    a2 = r^2;
}
    """

    def __init__(self, freq, t60, gain, trainable=False):
        super().__init__()
        self.trainable = trainable
        if not trainable:
            self.b0 = 1
            self.b1 = 0
            self.b2 = -1
            w = 2 * np.pi * freq / SR
            r = pow(0.001, 1 / float(t60 * SR))
            self.a1 = -2 * r * np.cos(w)
            self.a2 = np.power(r, 2)
            self.gain = gain

    def forward(self, x, **kwargs):
        if not self.trainable:
            a_coeff = torch.tensor([1, self.a1, self.a2])
            b_coeff = torch.tensor([self.b0, self.b1, self.b2])
        else:
            assert 'a_coeff' in kwargs and 'b_coeff' in kwargs, "a_coeff and b_coeff should be provided when trainable is True"
            a_coeff, b_coeff = kwargs['a_coeff'], kwargs['b_coeff']

        return self.gain * torchaudio.functional.lfilter(x, a_coeff, b_coeff)
