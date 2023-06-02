import torch.nn as nn


class TemporalShift(nn.Module):
    def __init__(self, n_segment=3, n_div=8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        out = x.clone().zero_()
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]  # shift right
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]  # not shift

        return out.view(nt, c, h, w)
