import torch
import torch.nn as nn

from Core import get_filter_params, get_reals_and_imgs


# from GuitarString import GuitarString


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


class TrainableLoss(nn.Module):
    def __init__(self, langrange_n):
        super().__init__()
        self.langrange_n = langrange_n

    def forward(self, length, pluckposition):
        nUp = length * pluckposition
        nDown = length * (1 - pluckposition)
        o = (self.langrange_n - 1.00001) / 2
        loss = torch.relu(o - nUp) + torch.relu(o - nDown)
        # print(f'nUp: {nUp}, nDown: {nDown}, o: {o}, Loss: {loss}')
        return loss


class PretrainLoss(nn.Module):
    def __init__(self, lambda_penalty, langrange_n):
        super().__init__()
        self.stability_loss = StabilityLoss(lambda_penalty)
        self.trainable_loss = TrainableLoss(langrange_n)

    def forward(self, length, pluckposition, filter_params):
        nut_params, bridge_params, dispersion_params = get_filter_params(filter_params)
        nut_a_reals, nut_a_imgs, nut_b_reals, nut_b_imgs = get_reals_and_imgs(nut_params)
        bridge_a_reals, bridge_a_imgs, bridge_b_reals, bridge_b_imgs = get_reals_and_imgs(bridge_params)
        dispersion_a_reals, dispersion_a_imgs, dispersion_b_reals, dispersion_b_imgs = get_reals_and_imgs(
            dispersion_params)

        loss = self.stability_loss(nut_a_reals, nut_a_imgs) + self.stability_loss(bridge_a_reals,
                                                                                  bridge_a_imgs) + self.stability_loss(
            dispersion_a_reals, dispersion_a_imgs) + self.trainable_loss(length, pluckposition)

        return loss

# if __name__ == "__main__":
#
#     # order = 2
#     #
#     # nut_a_reals = torch.randn(order, requires_grad=True)
#     # nut_a_imgs = torch.randn(order, requires_grad=True)
#     # nut_b_reals = torch.randn(order, requires_grad=True)
#     # nut_b_imgs = torch.randn(order, requires_grad=True)
#     #
#     # bridge_a_reals = torch.randn(order, requires_grad=True)
#     # bridge_a_imgs = torch.randn(order, requires_grad=True)
#     # bridge_b_reals = torch.randn(order, requires_grad=True)
#     # bridge_b_imgs = torch.randn(order, requires_grad=True)
#     #
#     # dispersion_a_reals = torch.randn(order, requires_grad=True)
#     # dispersion_a_imgs = torch.randn(order, requires_grad=True)
#     # dispersion_b_reals = torch.randn(order, requires_grad=True)
#     # dispersion_b_imgs = torch.randn(order, requires_grad=True)
#     #
#     # stability_loss = StabilityLoss(0.1)
#     #
#     # optim = torch.optim.Adam(
#     #     [nut_a_reals, nut_a_imgs, bridge_a_reals, bridge_a_imgs, dispersion_a_reals, dispersion_a_imgs], lr=0.01)
#     #
#     # for i in range(300):
#     #     optim.zero_grad()
#     #     loss = stability_loss(nut_a_reals, nut_a_imgs) + stability_loss(bridge_a_reals, bridge_a_imgs) + stability_loss(
#     #         dispersion_a_reals, dispersion_a_imgs)
#     #
#     #     loss.backward()
#     #     optim.step()
#     #     if loss.item() == 0:
#     #         break
#     #
#     # nut_poles_complex = get_complex(nut_a_reals, nut_a_imgs)
#     # nut_a_coeffs = get_coeffs_from_roots(nut_poles_complex)
#     #
#     # nut_b_zeros_complex = get_complex(nut_b_reals, nut_b_imgs)
#     # nut_b_coeffs = get_coeffs_from_roots(nut_b_zeros_complex)
#     #
#     # nut_b_coeffs = coeffs_norm(nut_b_coeffs, nut_a_coeffs)
#     # print(nut_b_coeffs)
#     # print(nut_a_coeffs)
#     #
#     # ar_nut = get_freq_response(nut_b_coeffs, nut_a_coeffs).abs()
#     # plt.plot(ar_nut.detach().numpy())
#     #
#     # bridge_poles_complex = get_complex(bridge_a_reals, bridge_a_imgs)
#     # bridge_a_coeffs = get_coeffs_from_roots(bridge_poles_complex)
#     #
#     # bridge_b_zeros_complex = get_complex(bridge_b_reals, bridge_b_imgs)
#     # bridge_b_coeffs = get_coeffs_from_roots(bridge_b_zeros_complex)
#     #
#     # bridge_b_coeffs = coeffs_norm(bridge_b_coeffs, bridge_a_coeffs)
#     # print(bridge_b_coeffs)
#     # print(bridge_a_coeffs)
#     #
#     # ar_bridge = get_freq_response(bridge_b_coeffs, bridge_a_coeffs).abs()
#     # plt.plot(ar_bridge.detach().numpy())
#     #
#     # dispersion_poles_complex = get_complex(dispersion_a_reals, dispersion_a_imgs)
#     # dispersion_a_coeffs = get_coeffs_from_roots(dispersion_poles_complex)
#     #
#     # dispersion_b_zeros_complex = get_complex(dispersion_b_reals, dispersion_b_imgs)
#     # dispersion_b_coeffs = get_coeffs_from_roots(dispersion_b_zeros_complex)
#     #
#     # dispersion_b_coeffs = coeffs_norm(dispersion_b_coeffs, dispersion_a_coeffs)
#     # print(dispersion_b_coeffs)
#     # print(dispersion_a_coeffs)
#     #
#     # ar_dispersion = get_freq_response(dispersion_b_coeffs, dispersion_a_coeffs).abs()
#     # plt.plot(ar_dispersion.detach().numpy())
#     # plt.show()
#     #
#     # guitar = GuitarString(seconds=2, trainable=False)
#     #
#     # excitation = torch.zeros(2 * 16000)
#     # excitation[0] = 0.5
#     #
#     # bridge_params = torch.stack([bridge_b_coeffs, bridge_a_coeffs])
#     # nuts_params = torch.stack([nut_b_coeffs, nut_a_coeffs])
#     # dispersion_params = torch.stack([dispersion_b_coeffs, dispersion_a_coeffs])
#     #
#     # output = guitar.forward(30.2,
#     #                         0.5,
#     #                         excitation,
#     #                         bridge_params=bridge_params,
#     #                         nuts_params=bridge_params,
#     #                         dispersion_params=bridge_params,
#     #                         )
#     #
#     # plt.plot(output.cpu().detach().numpy())
#     # plt.show()
#     #
#     # torchaudio.save("output.wav", output.cpu().detach().unsqueeze(0), 16000)
