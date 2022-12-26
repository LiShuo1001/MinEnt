import torch
from torch.distributions import Categorical


class Stat(object):

    @staticmethod
    def stat_mean(feature):
        return feature.mean(0).abs().mean()

    @staticmethod
    def stat_std(feature):
        return feature.std(0).mean()

    @staticmethod
    def stat_ent(feature):
        return Categorical(probs=torch.softmax(feature, dim=1)).entropy().mean()

    pass


class CompareMethod(object):
    MinEnt = "MinEntMultiHead"
    MinEntMultiHead = "MinEntMultiHead"
    MinEntOneHead = "MinEntOneHead"
    pass


class DatasetName(object):
    cifar10 = "cifar10"
    cifar100 = "cifar100"
    stl10 = "stl10"
    tiny = "tiny"
    imagenet100 = "imagenet100"
    pass


class NormName(object):
    bn = "BN"
    softmax = "softmax"
    l2norm = "l2norm"
    no = "no"
    pass


class OptimName(object):
    sgd = "sgd"
    adam = "adam"
    pass
