import crepe
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Constant import SR
from Core import f2l


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[..., :-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=out_channels // 2, out_channels=out_channels,
                         kernel_size=2 * stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class PitchExtractor(nn.Module):
    def __init__(self,
                 sample_rate: int,
                 hop_length: int,
                 ):
        """
        The PitchExtractor extract pitch (f0) from the audio signal.

        Shape:
             - Input: audio signal, with shape (Batch_size, Sample_length)
             - Output: mean amplitude, with shape (Batch_size, Timeframe), where Timeframe = `1 + Sample length // hop_length`

        Args:
            sample_rate (int): The sample rate of the audio signal.
            hop_length (int): The hop length for the STFT.

        """
        super().__init__()
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        # Convert the signal to numpy format as crepe only accepts numpy arrays
        signals = signals.numpy()
        batch_size = signals.shape[0]

        f0_batch = []
        # Use crepe to predict the pitch (f0) of the signal
        step_size = int(1000 * self.hop_length / self.sample_rate)

        for batch_index in range(batch_size):
            time, f0, _, _ = crepe.predict(
                signals[batch_index],
                sr=self.sample_rate,
                step_size=step_size,
                verbose=0,
                center=True,
                viterbi=True,
            )
            f0_batch.append(f0)
        f0_batch = np.array(f0_batch)
        f0 = torch.tensor(f0_batch, dtype=torch.float32)

        # f0.shape = [Batch_size, Timeframe]
        return f0


class LatentEncoder(nn.Module):
    def __init__(self, C, D, n_filter_params=2, signal_length=64000):
        super().__init__()
        self.pitchextractor = PitchExtractor(SR, 320)
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            nn.ELU(),
            EncoderBlock(out_channels=2 * C, stride=2),
            nn.ELU(),
            EncoderBlock(out_channels=4 * C, stride=4),
            nn.ELU(),
            EncoderBlock(out_channels=8 * C, stride=5),
            nn.ELU(),
            EncoderBlock(out_channels=16 * C, stride=8),
            nn.ELU(),
            CausalConv1d(in_channels=16 * C, out_channels=D, kernel_size=3)
        )

        self.last_layer = nn.Linear(int(D * (signal_length / 320)), 3 * (n_filter_params * 2) + 1)

    def forward(self, x):
        pitch = self.pitchextractor(x)
        length = f2l(pitch.mean(dim=0))
        batchsize = x.shape[0]
        x2 = self.layers(x)
        x2 = x2.view(batchsize, -1)
        latent = self.last_layer(x2)
        pluckposition = latent[:, 0]
        filter_params = latent[:, 1:]
        return length, F.sigmoid(pluckposition), filter_params


if __name__ == "__main__":
    encoder = LatentEncoder(64, 3, 5, 64000)
    x = torch.randn(1, 1, 64000)
    length, pluckposition, filter_params = encoder(x)

    # latent should contain: nut + bridge + dispersion shape is [batch, 3, 4

    # filter has a_real, a_imag, b_real, b_imag shape is [batch, 3, 4, order]
    # pluckposition is [batch, 3, 1]
    # plucklength is [batch, 3, 1]
