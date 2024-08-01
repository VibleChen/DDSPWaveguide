import torch
import torch.nn as nn
import torchaudio
from matplotlib import pyplot as plt

from Core import get_complex, get_coeffs_from_roots, get_freq_response


class StabilityLoss(nn.Module):
    def __init__(self, lambda_penalty):
        super().__init__()
        self.lambda_penalty = torch.tensor(lambda_penalty, dtype=torch.float32, requires_grad=True)

    def forward(self, reals, imgs):
        # 计算 a^2 + b^2
        constraint = reals ** 2 + imgs ** 2

        # 计算惩罚项，确保惩罚项是计算图的一部分
        penalty = torch.relu(constraint - 1)

        # 计算总损失
        total_loss = self.lambda_penalty * penalty

        return total_loss.sum()


def coeffs_norm(b_coeffs, a_coffs):
    fr = get_freq_response(b_coeffs, a_coffs)
    ar = torch.abs(fr)

    return b_coeffs / ar.max()


if __name__ == "__main__":

    from GuitarString import GuitarString

    order = 2
    k = 1

    nut_a_reals = torch.randn(order, requires_grad=True)
    nut_a_imgs = torch.randn(order, requires_grad=True)
    nut_b_reals = torch.randn(order, requires_grad=True)
    nut_b_imgs = torch.randn(order, requires_grad=True)

    bridge_a_reals = torch.randn(order, requires_grad=True)
    bridge_a_imgs = torch.randn(order, requires_grad=True)
    bridge_b_reals = torch.randn(order, requires_grad=True)
    bridge_b_imgs = torch.randn(order, requires_grad=True)

    dispersion_a_reals = torch.randn(order, requires_grad=True)
    dispersion_a_imgs = torch.randn(order, requires_grad=True)
    dispersion_b_reals = torch.randn(order, requires_grad=True)
    dispersion_b_imgs = torch.randn(order, requires_grad=True)

    stability_loss = StabilityLoss(0.1)

    optim = torch.optim.Adam(
        [nut_a_reals, nut_a_imgs, bridge_a_reals, bridge_a_imgs, dispersion_a_reals, dispersion_a_imgs], lr=0.01)

    for i in range(300):
        optim.zero_grad()
        loss = stability_loss(nut_a_reals, nut_a_imgs) + stability_loss(bridge_a_reals, bridge_a_imgs) + stability_loss(
            dispersion_a_reals, dispersion_a_imgs)

        print(loss)
        loss.backward()
        optim.step()

    nut_poles_complex = get_complex(nut_a_reals, nut_a_imgs)
    nut_a_coeffs = get_coeffs_from_roots(nut_poles_complex)


    nut_b_zeros_complex = get_complex(nut_b_reals, nut_b_imgs)
    nut_b_coeffs = get_coeffs_from_roots(nut_b_zeros_complex)

    nut_b_coeffs = coeffs_norm(nut_b_coeffs, nut_a_coeffs) * k
    print(nut_b_coeffs)
    print(nut_a_coeffs)

    ar_nut = get_freq_response(nut_b_coeffs, nut_a_coeffs).abs()
    plt.plot(ar_nut.detach().numpy())

    bridge_poles_complex = get_complex(bridge_a_reals, bridge_a_imgs)
    bridge_a_coeffs = get_coeffs_from_roots(bridge_poles_complex)


    bridge_b_zeros_complex = get_complex(bridge_b_reals, bridge_b_imgs)
    bridge_b_coeffs = get_coeffs_from_roots(bridge_b_zeros_complex)

    bridge_b_coeffs = coeffs_norm(bridge_b_coeffs, bridge_a_coeffs) * k
    print(bridge_b_coeffs)
    print(bridge_a_coeffs)

    ar_bridge = get_freq_response(bridge_b_coeffs, bridge_a_coeffs).abs()
    plt.plot(ar_bridge.detach().numpy())

    dispersion_poles_complex = get_complex(dispersion_a_reals, dispersion_a_imgs)
    dispersion_a_coeffs = get_coeffs_from_roots(dispersion_poles_complex)


    dispersion_b_zeros_complex = get_complex(dispersion_b_reals, dispersion_b_imgs)
    dispersion_b_coeffs = get_coeffs_from_roots(dispersion_b_zeros_complex)

    dispersion_b_coeffs = coeffs_norm(dispersion_b_coeffs, dispersion_a_coeffs) * k
    print(dispersion_b_coeffs)
    print(dispersion_a_coeffs)

    ar_dispersion = get_freq_response(dispersion_b_coeffs, dispersion_a_coeffs).abs()
    plt.plot(ar_dispersion.detach().numpy())
    plt.show()

    guitar = GuitarString(seconds=2, trainable=True)

    excitation = torch.zeros(2 * 16000)
    excitation[0] = 0.5

    bridge_params = torch.stack([bridge_b_coeffs, bridge_a_coeffs])
    nuts_params = torch.stack([nut_b_coeffs, nut_a_coeffs])
    dispersion_params = torch.stack([dispersion_b_coeffs, dispersion_a_coeffs])

    output = guitar.forward(20.2,
                            0.5506,
                            excitation,
                            bridge_params=bridge_params,
                            nuts_params=nuts_params,
                            dispersion_params=dispersion_params,
                            )

    plt.plot(output.cpu().detach().numpy())
    plt.show()