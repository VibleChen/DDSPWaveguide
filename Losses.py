import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from Constant import device
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
        self.eps = 1e-6

    def forward(self, length, pluckposition):
        nUp = length * pluckposition
        nDown = length * (1 - pluckposition)
        o = (self.langrange_n - 1.00001) / 2
        loss = torch.relu(o - nUp + self.eps) + torch.relu(o - nDown + self.eps)
        # print(f'nUp: {nUp}, nDown: {nDown}, o: {o}, Loss: {loss}')
        return loss


class PretrainLoss(nn.Module):
    def __init__(self, lambda_penalty=0.1, langrange_n=4):
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


class MultiSpectralLoss(nn.Module):
    """Multi-Scale spectrogram loss.

      This loss is the bread-and-butter of comparing two audio signals. It offers
      a range of options to compare spectrograms, many of which are redunant, but
      emphasize different aspects of the signal. By far, the most common comparisons
      are magnitudes (mag_weight) and log magnitudes (logmag_weight).

      Shape:
        - Input: generated and target audio signal, with both shape (Batch_size, Sample_length)
        - Output: computed loss, a scalar value

      Args:
        fft_sizes (tuple): A tuple of FFT sizes to use for the STFT. Default is (2048, 1024, 512, 256, 128, 64).
        loss_type (str): The type of loss to use ('L1', 'L2', or 'COSINE'). Default is 'L1'.
        mag_weight (float): Weight for the magnitude loss. Default is 1.0.
        delta_time_weight (float): Weight for the delta time loss. Default is 0.0.
        delta_freq_weight (float): Weight for the delta frequency loss. Default is 0.0.
        cumsum_freq_weight (float): Weight for the cumulative sum frequency loss. Default is 0.0.
        logmag_weight (float): Weight for the log magnitude loss. Default is 0.0.
        loudness_weight (float): Weight for the loudness loss. Default is 0.0.
        """

    def __init__(self,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 loss_type='L1',
                 mag_weight=1.0,
                 delta_time_weight=0.0,
                 delta_freq_weight=0.0,
                 cumsum_freq_weight=0.0,
                 logmag_weight=0.0,
                 loudness_weight=0.0,
                 ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.delta_time_weight = delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.cumsum_freq_weight = cumsum_freq_weight
        self.logmag_weight = logmag_weight
        self.loudness_weight = loudness_weight
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, target_audio, audio):
        loss = 0.0

        for size in self.fft_sizes:
            target_mag = T.Spectrogram(n_fft=size).to(device)(target_audio)
            value_mag = T.Spectrogram(n_fft=size).to(device)(audio)

            if self.mag_weight > 0:
                if self.loss_type == 'L1':
                    loss += self.mag_weight * F.l1_loss(target_mag, value_mag, reduction='mean')
                elif self.loss_type == 'L2':
                    loss += self.mag_weight * F.mse_loss(target_mag, value_mag, reduction='mean')
                elif self.loss_type == 'COSINE':
                    loss += self.mag_weight * (1 - self.cosine_similarity(target_mag, value_mag).mean())

            if self.delta_time_weight > 0:
                target = torch.diff(target_mag, dim=1)
                value = torch.diff(value_mag, dim=1)
                if self.loss_type == 'L1':
                    loss += self.delta_time_weight * F.l1_loss(target, value, reduction='mean')
                elif self.loss_type == 'L2':
                    loss += self.delta_time_weight * F.mse_loss(target, value, reduction='mean')
                elif self.loss_type == 'COSINE':
                    loss += self.mag_weight * (1 - self.cosine_similarity(target, value).mean())

            if self.delta_freq_weight > 0:
                target = torch.diff(target_mag, dim=2)
                value = torch.diff(value_mag, dim=2)
                if self.loss_type == 'L1':
                    loss += self.delta_freq_weight * F.l1_loss(target, value, reduction='mean')
                elif self.loss_type == 'L2':
                    loss += self.delta_freq_weight * F.mse_loss(target, value, reduction='mean')
                elif self.loss_type == 'COSINE':
                    loss += self.mag_weight * (1 - self.cosine_similarity(target, value).mean())

            if self.cumsum_freq_weight > 0:
                target = torch.cumsum(target_mag, dim=2)
                value = torch.cumsum(value_mag, dim=2)
                if self.loss_type == 'L1':
                    loss += self.cumsum_freq_weight * F.l1_loss(target, value, reduction='mean')
                elif self.loss_type == 'L2':
                    loss += self.cumsum_freq_weight * F.mse_loss(target, value, reduction='mean')
                elif self.loss_type == 'COSINE':
                    loss += self.mag_weight * (1 - self.cosine_similarity(target, value).mean())

            if self.logmag_weight > 0:
                target = torch.log(target_mag)
                value = torch.log(value_mag)
                if self.loss_type == 'L1':
                    loss += self.logmag_weight * F.l1_loss(target, value, reduction='mean')
                elif self.loss_type == 'L2':
                    loss += self.logmag_weight * F.mse_loss(target, value, reduction='mean')
                elif self.loss_type == 'COSINE':
                    loss += self.mag_weight * (1 - self.cosine_similarity(target, value).mean())

        if self.loudness_weight > 0:
            target = T.AmplitudeToDB()(target_audio)
            value = T.AmplitudeToDB()(audio)
            if self.loss_type == 'L1':
                loss += self.loudness_weight * F.l1_loss(target, value, reduction='mean')
            elif self.loss_type == 'L2':
                loss += self.loudness_weight * F.mse_loss(target, value, reduction='mean')
            elif self.loss_type == 'COSINE':
                loss += self.mag_weight * (1 - self.cosine_similarity(target, value).mean())

        return loss
