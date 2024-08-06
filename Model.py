import torch
import torch.nn as nn

from Constant import SR
from Core import get_filters_coeffs
from Encoder import LatentEncoder
from GuitarString import GuitarString


def get_excitation(batch_size, signal_length):
    excitation = torch.zeros((batch_size, signal_length), dtype=torch.float32)
    excitation[:, 0] = 0.5
    return excitation


class DDSPEncoderDecoderModel(nn.Module):
    def __init__(self, batch_size=1, C=64, D=3, n_filter_params=10, signal_length=64000, trainable=True):
        super().__init__()

        self.encoder = LatentEncoder(C, D, n_filter_params, signal_length)
        self.decoder = GuitarString(batch_size=batch_size, seconds=signal_length / SR, trainable=trainable)
        self.excitation = get_excitation(batch_size, signal_length)

    def forward(self, x=None, length=None, pluckposition=None, filter_params=None, strategy='encoder'):
        if strategy == 'encoder':
            return self.encoder(x)
        elif strategy == 'decoder':
            nut_a_coeffs, nut_b_coeffs, bridge_a_coeffs, bridge_b_coeffs, dispersion_a_coeffs, dispersion_b_coeffs = get_filters_coeffs(
                filter_params)
            return self.decoder(length, pluckposition,
                                excitation=self.excitation,
                                bridge_params=(bridge_b_coeffs, bridge_a_coeffs),
                                nuts_params=(nut_b_coeffs, nut_a_coeffs),
                                dispersion_params=(dispersion_b_coeffs, dispersion_a_coeffs))
