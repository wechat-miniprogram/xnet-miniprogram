import torch
import torch.nn.functional as F
import numpy as np
from visdom import Visdom

_CURRENT_PLOTTER = None


class VisdomPlotter(object):
    def __init__(self, port):
        self.viz = Visdom(port=port)
        self.bin = 256
        self.visited = {}
        self.enable_plotter = True

    def plot(self, data, tag):
        if not self.enable_plotter:
            return

        if "activation_quantizers" in tag:
            # batch_size = data.shape[0]
            flatten_data = data.flatten()
            histogram = torch.histc(flatten_data, bins=self.bin)
            self.viz.bar(X=histogram, win=tag, opts=dict(title=tag), env="stats")
        elif "weight_fake_quant" in tag:
            flatten_data = data.flatten()
            histogram = torch.histc(flatten_data, bins=self.bin)
            self.viz.bar(X=histogram, win=tag, opts=dict(title=tag), env="stats")
        else:
            assert False, tag

    def plot_dc(self, data, fq_data, tag):
        if not self.enable_plotter:
            return

        flatten_data = data.flatten()
        flatten_fq_data = fq_data.flatten()
        dc = 1.0 - F.cosine_similarity(flatten_data, flatten_fq_data, 0, 1e-8)
        Y = dc.detach().cpu().numpy().reshape(1)
        if tag not in self.visited:
            self.visited[tag] = 0
            index = self.visited[tag]
            self.viz.line(
                X=np.array([index]), Y=Y, win=tag, env="dc", opts=dict(title=tag)
            )
        else:
            self.visited[tag] += 1
            index = self.visited[tag]
            self.viz.line(
                X=np.array([index]),
                Y=Y,
                win=tag,
                env="dc",
                opts=dict(title=tag),
                update="append",
            )

    def enable(self, _enable=True):
        self.enable_plotter = _enable

    def disable(self):
        self.enable(_enable=False)


# _CURRENT_PLOTTER = VisdomPlotter(8001)
