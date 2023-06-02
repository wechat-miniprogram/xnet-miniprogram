import torch
import torch.nn as nn


class MasterFakeQuantize(nn.Module):
    def __init__(self):
        super(MasterFakeQuantize, self).__init__()
        self.fq_list = None
        self.master_fq = None

    def forward(self, X):
        if self.training:
            Y = self.master_fq(X)
            self.broadcast_qparam()
            return Y
        else:
            Y = self.master_fq(X)

        return Y

    def broadcast_qparam(self):
        for fq in self.fq_list:
            if isinstance(fq, MasterFakeQuantize):
                fq.master_fq.close_observer()
                fq.master_fq.scale.copy_(self.master_fq.scale)
                fq.master_fq.zero_point.copy_(self.master_fq.zero_point)
                fq.broadcast_qparam()
            else:
                fq.close_observer()
                fq.scale.copy_(self.master_fq.scale)
                fq.zero_point.copy_(self.master_fq.zero_point)


class QConcat(nn.Module):
    def __init__(self):
        super(QConcat, self).__init__()
        self.master_fq = MasterFakeQuantize()

    def forward(self, X, dim=0):
        Y = torch.cat(X, dim=dim)
        Y_fq = self.master_fq(Y)
        return Y_fq
