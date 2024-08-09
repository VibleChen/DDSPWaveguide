import torch
import torch.nn as nn

from Constant import SR, device
from Modules import _Chain, GuitarNuts, GuitarBridge, StringSegment, DispersionFilter, In


class GuitarString(nn.Module):
    """
    GuitarModel(length,pluckPosition,mute,excitation) = endChain(egChain)
with{
    maxStringLength = maxLength;
    lengthTuning = 0.11; // tuned "by hand"
    stringL = length-lengthTuning;
    muteBlock = *(mute),*(mute),_;
    egChain = chain(
        GuitarNuts :
        openStringPick(stringL,0.05,pluckPosition,excitation) :
        muteBlock :
        GuitarBridge);
};

    """

    def __init__(self,
                 batch_size=1,
                 trainable=False,
                 seconds=1.0,
                 ):
        super().__init__()

        self.trainable = trainable
        # self.zeros = torch.zeros((batch_size, int(seconds * SR)), dtype=torch.float32).to(device)
        self.zeros = torch.zeros(int(seconds * SR), dtype=torch.float).to(device)
        self.nuts = GuitarNuts(trainable=trainable)
        self.bridge = GuitarBridge(trainable=trainable)
        self.dispersionfilter = DispersionFilter(trainable=trainable)

    def _get_chain(self, nUp, nDown, exciation):
        chain = _Chain(
            self.nuts,
            StringSegment(nUp),
            In(exciation),
            self.dispersionfilter,
            StringSegment(nDown),
            self.bridge,
        )

        return chain

    def forward(self, length, pluckPosition, excitation, **kwargs):
        """
        length (float or int): the delay line in samples
        pluckposition (float): a float number between 0 and 1
        excitation (tensor): the excitation for initialize. It has to be the same length with output length
        """

        nUp = length * pluckPosition
        nDown = length * (1 - pluckPosition)

        chain = self._get_chain(nUp, nDown, excitation)

        # initialize the left and right signals
        left, right = chain._start(self.zeros, self.zeros, **kwargs)

        n_loops = excitation.size(-1) // int(length.min())

        left_signal = torch.zeros((n_loops + 1, *left.shape), dtype=left.dtype, device=left.device)
        right_signal = torch.zeros((n_loops + 1, *right.shape), dtype=right.dtype, device=right.device)

        left_signal[0] = left
        right_signal[0] = right

        for i in range(n_loops):
            right_signal[i + 1] = chain._lterminate(left_signal[i], self.zeros, **kwargs)
            left_signal[i + 1] = chain._rterminate(self.zeros, right_signal[i], **kwargs)
            left_signal[i + 1], right_signal[i + 1] = chain._vibrate(left_signal[i + 1], right_signal[i + 1], **kwargs)

        return torch.sum(left_signal, dim=0), torch.sum(right_signal, dim=0)