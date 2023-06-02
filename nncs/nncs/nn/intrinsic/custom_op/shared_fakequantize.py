import copy
import torch.nn as nn


class SharedFakeQuantize(nn.Module):
    def __init__(self, fq):
        super(SharedFakeQuantize, self).__init__()
        self.fq = fq

    def forward(self, X):
        if self.training:
            is_ob = self.fq.observer_enabled[0]
            self.fq.disable_observer()
            if isinstance(X, list) or isinstance(X, tuple):
                Ys = []
                for _X in X:
                    _Y = self.fq(_X)
                    Ys.append(_Y)
                Ys = tuple(Ys)
            else:
                Ys = self.fq(X)
            self.fq.observer_enabled[0] = is_ob
            return Ys
        else:
            if isinstance(X, list) or isinstance(X, tuple):
                Ys = []
                for _X in X:
                    _Y = self.fq(_X)
                    Ys.append(_Y)
                Ys = tuple(Ys)
            else:
                Ys = self.fq(X)

        return Ys


class MasterFakeQuantize(nn.Module):
    def __init__(self, fq_list):
        super(MasterFakeQuantize, self).__init__()
        self.fq_list = fq_list
        self.master_fq = copy.deepcopy(self.fq_list[0])
        self.is_first = True

    def forward(self, X):
        if self.training:
            if self.is_first:
                Y = self.master_fq(X)

                for fq in self.fq_list:
                    fq.disable_observer()
                    fq.scale.copy_(self.master_fq.scale)
                    fq.zero_point.copy_(self.master_fq.zero_point)

                self.is_first = False
            else:
                Y = self.master_fq(X)
                for fq in self.fq_list:
                    fq.scale.copy_(self.master_fq.scale)
                    fq.zero_point.copy_(self.master_fq.zero_point)
            return Y
        else:
            Y = self.master_fq(X)

        return Y
