import torch
import torch.nn as nn
import torch.nn.functional as F

from Constant import SR


class FeedbackDelayNetwork(nn.Module):
    """
    Differentiable Feedback Delay Network reverb.
    Simplified from https://github.com/phvial/priv-ddfx/blob/main/effects.py
    (credits: Pierre-Hugo Vial, 2023, AQUA-RIUS ANR project).
    Config fixed to:
    - decorrelator: allpass
    - mixing_matrix_style: Householder
    - reverb_time_control: onepole
    - early_rev_style: fir
    - delay_lines: 8
    - VNS_input: False
    - VNS_output: False
    """

    def __init__(
            self,
            sampling_rate=SR,
            delay_lines=8,
            early_ir_length=200,
            early_reflections=6,
            time_control_bands=6,
    ):
        """Initializes DelayNetwork object.

        Parameters
        ----------
        sampling_rate : float, optional
            sampling rate, by default SR
        delay_lines : int, optional
            number of delay lines, by default 6
        early_ir_length : int, optional
            length of IR for early reflections, by default 200
        early_reflections : int, optional
            number of early reflections with early_rev_style!='fir, by default 6
        time_control_bands : int, optional
            number of bands for reverberation time control, by default 6
        """
        super().__init__()
        self.sampling_rate = torch.tensor(sampling_rate, dtype=torch.float32)
        self.freq_points = int(2 * self.sampling_rate)
        self.early_ir_length = early_ir_length
        self.delay_lines = delay_lines
        self.early_reflections = early_reflections
        self.time_control_bands = time_control_bands

        """Builds DelayNetwork object."""
        # ---- builds mixing matrix
        self.mixing_matrix = -1 * torch.eye(self.delay_lines) + 0.5 * torch.ones([self.delay_lines, self.delay_lines])
        # ---- builds trainable variables
        general_initializer = self._random_normal_initializer(mean=0.0, std=1e-1)
        delay_initializer = self._random_normal_initializer(mean=400.0, std=60.0)
        gains_initializer = self._random_normal_initializer(mean=0.25, std=1e-1)

        self.early_ir = nn.Parameter(general_initializer(torch.empty(self.early_ir_length)))
        self.input_gain = nn.Parameter(gains_initializer(torch.empty(self.delay_lines)))
        self.output_gain = nn.Parameter(gains_initializer(torch.empty(self.delay_lines)))
        self.time_rev_0_sec = nn.Parameter(general_initializer(torch.empty(1)).clamp(min=0.0))
        self.alpha_tone = nn.Parameter(general_initializer(torch.empty(1)))
        self.delay_values = nn.Parameter(delay_initializer(torch.empty(self.delay_lines)))
        self.delays_allpass = nn.Parameter(delay_initializer(torch.empty(self.delay_lines, 4)))
        self.gain_allpass = nn.Parameter(gains_initializer(torch.empty(self.delay_lines, 4)))

    def __len__(self):
        return self.delay_lines

    def _random_normal_initializer(self, mean=0.0, std=1e-1):
        def _initializer(tensor):
            return nn.init.normal_(tensor, mean=mean, std=std)

        return _initializer

    def get_late_ir(self):
        """Returns IR for late reverberation."""

        input_gain = torch.complex(self.input_gain, torch.tensor(0.0))
        output_gain = torch.complex(self.output_gain, torch.tensor(0.0))
        mixing_matrix = torch.complex(self.mixing_matrix, torch.tensor(0.0))

        if len(mixing_matrix.shape) == 2:
            mixing_matrix = mixing_matrix.unsqueeze(0).repeat(self.freq_points // 2 + 1, 1, 1)
        if len(output_gain.shape) == 1:
            output_gain = output_gain.unsqueeze(0).unsqueeze(0).repeat(self.freq_points // 2 + 1, 1, 1)
        if len(input_gain.shape) == 1:
            input_gain = input_gain.unsqueeze(1).unsqueeze(0).repeat(self.freq_points // 2 + 1, 1, 1)
        eye_mat = torch.eye(mixing_matrix.shape[-1], dtype=torch.complex64).unsqueeze(0).repeat(
            self.freq_points // 2 + 1, 1, 1)

        # generate normalized frequencies vector
        wk = torch.complex(
            (2 * torch.pi * torch.arange(self.freq_points // 2 + 1, dtype=torch.float32) / self.freq_points),
            torch.tensor(0.0))  # shape: [freq_points//2+1]

        z_d = torch.stack(
            [
                torch.exp(-1j * wk * torch.floor(self.delay_values[d]))
                for d in range(self.delay_lines)
            ],
            dim=1,
        )  # shape:  [freq_points//2+1, delay_lines], matrix of z^{-d}, elements of D

        d_eta = self.delay_values - torch.floor(self.delay_values)
        eta = (1 - d_eta) / (1 + d_eta)
        allpass_interp = torch.stack(
            [
                (eta[d] + torch.exp(-1j * wk)) / (1 + eta[d] * torch.exp(-1j * wk))
                for d in range(self.delay_lines)
            ],
            dim=1,
        )

        diag_delay_matrix = torch.diag_embed(z_d * allpass_interp)

        delay_sec = (
                        self.delay_values + torch.sum(self.delays_allpass, dim=-1)
                    ) / self.sampling_rate

        # ---- (if) Onepole LP filter reverb time control
        k = torch.pow(10.0, -3 * delay_sec / self.time_rev_0_sec).unsqueeze(0)
        kpi = torch.pow(10.0, -3 * delay_sec / (self.alpha_tone * self.time_rev_0_sec)).unsqueeze(0)
        g = 2 * k * kpi / (k + kpi)
        p = (k - kpi) / (k + kpi)

        g_tiled = g.repeat(self.freq_points // 2 + 1, 1)
        p_tiled = p.repeat(self.freq_points // 2 + 1, 1)
        z_tiled = torch.exp(-1j * wk).unsqueeze(-1).repeat(1, self.delay_lines)

        sampled_transfers = g_tiled / (1 - p_tiled * z_tiled + 1e-8)
        filter_matrix = torch.diag_embed(sampled_transfers)

        gain_allpass = self.gain_allpass.unsqueeze(0).repeat(self.freq_points // 2 + 1, 1, 1)
        delays_allpass = self.delays_allpass.unsqueeze(0).repeat(self.freq_points // 2 + 1, 1, 1)
        wk_tiled = wk.unsqueeze(-1).unsqueeze(-1).repeat(1, delays_allpass.shape[1], delays_allpass.shape[2])

        z_delays = torch.exp(1j * wk_tiled * delays_allpass)

        allpass_transfer = torch.prod(
            (1 + gain_allpass * z_delays) / (gain_allpass + z_delays),
            dim=-1,
        )
        allpass_transfer_diag = torch.diag_embed(allpass_transfer)

        if len(filter_matrix.shape) == 4:
            mixing_matrix = mixing_matrix.unsqueeze(1)
        feedback_matrix = torch.matmul(
            torch.matmul(filter_matrix, mixing_matrix), allpass_transfer_diag
        )

        if len(feedback_matrix.shape) == 4:
            eye_mat = eye_mat.unsqueeze(1)
            diag_delay_matrix = diag_delay_matrix.unsqueeze(1)
            input_gain = input_gain.unsqueeze(0)
            output_gain = output_gain.unsqueeze(0)

        ir = torch.fft.irfft(
            torch.squeeze(
                torch.matmul(
                    output_gain,
                    torch.matmul(
                        torch.matmul(
                            diag_delay_matrix,
                            torch.linalg.inv(
                                eye_mat - torch.matmul(feedback_matrix, diag_delay_matrix)
                            ),
                        ),
                        input_gain,
                    ),
                )
            )
        )
        return ir

    def get_ir(self):
        """Returns reverb IR."""
        late_ir = self.get_late_ir()

        early_ir = self.early_ir.squeeze()
        if late_ir.shape[0] > early_ir.shape[0]:
            early_ir = F.pad(early_ir, (0, late_ir.shape[0] - early_ir.shape[0]))
        return early_ir[: late_ir.shape[0]] + late_ir

    def get_signal(self, audio: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        ir = ir.unsqueeze(0).unsqueeze(0)
        audio_out = torch.nn.functional.conv1d(audio, ir, padding='same')
        return audio_out

    def forward(self, audio_dry: torch.Tensor):

        ir = self.get_ir()
        audio_out = self.get_signal(audio_dry, ir)
        audio_out = audio_out.squeeze(1)
        return audio_out
