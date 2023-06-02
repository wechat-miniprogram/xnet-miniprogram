from easydict import EasyDict


class NNCSConfig(EasyDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
