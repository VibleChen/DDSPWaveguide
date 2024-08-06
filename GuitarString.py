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
        self.zeros = torch.zeros((batch_size, int(seconds * SR)), dtype=torch.float32).to(device)

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

        left_signal = [left]
        right_signal = [right]
        for i in range(excitation.size(-1) // int(length.min())):
            right_signal.append(chain._lterminate(left_signal[i], self.zeros, **kwargs))
            left_signal.append(chain._rterminate(self.zeros, right_signal[i], **kwargs))
            left_signal[-1], right_signal[-1] = chain._vibrate(left_signal[-1], right_signal[-1], **kwargs)

        return torch.sum(torch.stack(left_signal, dim=0), dim=0), torch.sum(torch.stack(right_signal, dim=0), dim=0)
