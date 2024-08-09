import abc

import torch
import torch.nn as nn
import torchaudio

from Constant import Langrange_N
from Delay import fdelayltv, delay
from Filters import BridgeFilter


class ModelBlock(nn.Module):

    def forward(self, left, right):
        return self.LeftGoingWaves(left, right), self.RightGoingWaves(left, right)

    @abc.abstractmethod
    def LeftGoingWaves(self, left, right, **kwargs):
        pass

    @abc.abstractmethod
    def RightGoingWaves(self, left, right, **kwargs):
        pass


class Emply(ModelBlock):

    def LeftGoingWaves(self, left, right, **kwargs):
        return left

    def RightGoingWaves(self, left, right, **kwargs):
        return right


class StringSegment(ModelBlock):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def LeftGoingWaves(self, left, right, **kwargs):
        return fdelayltv(Langrange_N, self.length, left)

    def RightGoingWaves(self, left, right, **kwargs):
        return fdelayltv(Langrange_N, self.length, right)


class _Chain:
    """
    核心类，用于连接各个模块
    """

    def __init__(self, *ModelBlock):
        self.blocks = ModelBlock
        self.instrument = tuple(block for block in self.blocks if
                                block.__class__.__name__ != "GuitarNuts" and block.__class__.__name__ != "GuitarBridge" and block.__class__.__name__ != "In")

    def _start(self, left, right, **kwargs):

        for i in range(len(self.blocks)):
            left = self.blocks[len(self.blocks) - (i + 1)].LeftGoingWaves(left, right, **kwargs)

            if i < len(self.blocks) - 1:
                left = delay(1, left)
                right = delay(1, right)

            right = self.blocks[i].RightGoingWaves(left, right, **kwargs)

        return left, right

    def _rterminate(self, left, right, **kwargs):

        left = self.blocks[len(self.blocks) - 1].LeftGoingWaves(left, right, **kwargs)

        return left

    def _lterminate(self, left, right, **kwargs):

        right = self.blocks[0].RightGoingWaves(left, right, **kwargs)

        return right

    def _vibrate(self, left, right, **kwargs):

        for i in range(len(self.instrument)):
            # 对于left是2，1，0的顺序
            left = self.instrument[len(self.instrument) - (i + 1)].LeftGoingWaves(left, right, **kwargs)
            left = delay(1, left)

            # 对于right，是1，2，3的顺序
            right = delay(1, right)
            right = self.instrument[i].RightGoingWaves(left, right, **kwargs)

        return left, right


class LeftTermination(ModelBlock):
    def __init__(self, Filter, block):
        super().__init__()
        self.ltermination_filter = Filter
        self.block = block

    def LeftGoingWaves(self, left, right, **kwargs):
        left = self.block.LeftGoingWaves(left, right, **kwargs)

        return left

    def RightGoingWaves(self, left, right, **kwargs):
        left = self.LeftGoingWaves(left, right, **kwargs)
        left = delay(1, left)
        left = self.ltermination_filter(left, **kwargs)

        right = self.block.RightGoingWaves(left, left, **kwargs)

        return right


class RightTermination(ModelBlock):
    def __init__(self, block, Filter):
        super().__init__()
        self.rtermination_filter = Filter
        self.block = block

    def LeftGoingWaves(self, left, right, **kwargs):
        right = self.RightGoingWaves(left, right, **kwargs)
        right = delay(1, right)
        right = self.rtermination_filter(right, **kwargs)

        left = self.block.LeftGoingWaves(right, right, **kwargs)
        left = delay(1, left)

        return left

    def RightGoingWaves(self, left, right, **kwargs):
        right = self.block.RightGoingWaves(left, right, **kwargs)

        return right


class InLeftWave(ModelBlock):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def LeftGoingWaves(self, left, right, **kwargs):
        return left + self.signal

    def RightGoingWaves(self, left, right, **kwargs):
        return right


class InRightWave(ModelBlock):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def LeftGoingWaves(self, left, right, **kwargs):
        return left

    def RightGoingWaves(self, left, right, **kwargs):
        return right + self.signal


class In(ModelBlock):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def LeftGoingWaves(self, left, right, **kwargs):
        return left + self.signal

    def RightGoingWaves(self, left, right, **kwargs):
        return right + self.signal


class DispersionFilter(ModelBlock):
    def __init__(self, stiffness=0.05, trainable=False):
        super().__init__()
        self.trainable = trainable
        if not trainable:
            self.stiffness = stiffness
            self.b_coeff = torch.tensor([1 - stiffness, 0])
            self.a_coeff = torch.tensor([1, -stiffness])

    def LeftGoingWaves(self, left, right, **kwargs):
        if not self.trainable:
            return torchaudio.functional.lfilter(left, self.a_coeff, self.b_coeff)
        else:
            assert 'dispersion_params' in kwargs, "Dispersion params should be provided when trainable is True"

            b_coeff, a_coeff = kwargs['dispersion_params']

            return torchaudio.functional.lfilter(left, a_coeff, b_coeff)

    def RightGoingWaves(self, left, right, **kwargs):
        if not self.trainable:
            return torchaudio.functional.lfilter(right, self.a_coeff, self.b_coeff)
        else:
            assert 'dispersion_params' in kwargs, "Dispersion params should be provided when trainable is True"
            b_coeff, a_coeff = kwargs['dispersion_params']

            return torchaudio.functional.lfilter(right, a_coeff, b_coeff)


# class GuitarNuts(LeftTermination):
#     """
#     guitarNuts = lTermination(-bridgeFilter(0.4, 0.5), basicBlock);
#     """
#
#     def __init__(self, brightness=0.8, absorption=0.6, trainable=False, block=Emply()):
#         filter = BridgeFilter(brightness, absorption, trainable=trainable)
#         super().__init__(filter, block)


class GuitarNuts(ModelBlock):
    def __init__(self, brightness=0.8, absorption=0.6, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.filter = BridgeFilter(brightness, absorption, trainable)
        self.block = Emply()

    def LeftGoingWaves(self, left, right, **kwargs):
        left = self.block.LeftGoingWaves(left, right, **kwargs)

        return left

    def RightGoingWaves(self, left, right, **kwargs):
        left = self.LeftGoingWaves(left, right, **kwargs)
        left = delay(1, left)
        if self.trainable:
            assert 'nuts_params' in kwargs, "Nuts params should be provided when trainable is True"
            b_coeff, a_coeff = kwargs['nuts_params']
        else:
            b_coeff, a_coeff = None, None

        left = self.filter(left, a_coeff, b_coeff)

        right = self.block.RightGoingWaves(left, left, **kwargs)

        return right


# class GuitarBridge(RightTermination):
#     def __init__(self, brightness=0.4, absorption=0.5, trainable=False, block=Emply()):
#         filter = BridgeFilter(brightness, absorption, trainable=trainable)
#         super().__init__(block, filter)


class GuitarBridge(ModelBlock):
    def __init__(self, brightness=0.4, absorption=0.5, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.filter = BridgeFilter(brightness, absorption, trainable)
        self.block = Emply()

    def LeftGoingWaves(self, left, right, **kwargs):
        right = self.RightGoingWaves(left, right, **kwargs)
        right = delay(1, right)
        if self.trainable:
            assert 'bridge_params' in kwargs, "Bridge params should be provided when trainable is True"
            b_coeff, a_coeff = kwargs['bridge_params']
        else:
            b_coeff, a_coeff = None, None

        right = self.filter(right, a_coeff, b_coeff)

        left = self.block.LeftGoingWaves(right, right, **kwargs)
        left = delay(1, left)

        return left

    def RightGoingWaves(self, left, right, **kwargs):
        right = self.block.RightGoingWaves(left, right, **kwargs)

        return right