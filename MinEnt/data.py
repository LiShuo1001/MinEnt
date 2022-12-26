import os
import random
from PIL import ImageFilter
from functools import lru_cache
import torchvision.transforms as T
from torch.utils.data import DataLoader
from abc import ABCMeta, abstractmethod
from torchvision.datasets import ImageFolder
from torchvision.datasets import STL10 as S10
from torchvision.datasets import CIFAR10 as C10
from torchvision.datasets import CIFAR100 as C100


def aug_transform_old(crop, base_transform, cfg, extra_t=[]):
    """ augmentation transform generated from config """
    return T.Compose(
        [
            T.RandomApply(
                [T.ColorJitter(cfg.cj0, cfg.cj1, cfg.cj2, cfg.cj3)], p=cfg.cj_p
            ),
            T.RandomGrayscale(p=cfg.gs_p),
            T.RandomResizedCrop(
                crop,
                scale=(cfg.crop_s0, cfg.crop_s1),
                ratio=(cfg.crop_r0, cfg.crop_r1),
                interpolation=3,
            ),
            T.RandomHorizontalFlip(p=cfg.hf_p),
            *extra_t,
            base_transform(),
        ]
    )


def aug_transform(crop, base_transform, cfg, extra_t=[]):
    trans_list = []
    if cfg.has_color_jitter:
        trans_list.append(T.RandomApply([T.ColorJitter(cfg.cj0, cfg.cj1, cfg.cj2, cfg.cj3)], p=cfg.cj_p))
    if cfg.has_gray_scale:
        trans_list.append(T.RandomGrayscale(p=cfg.gs_p))
    if cfg.has_crop:
        trans_list.append(T.RandomResizedCrop(crop, scale=(cfg.crop_s0, cfg.crop_s1),
                                              ratio=(cfg.crop_r0, cfg.crop_r1),interpolation=3))
    else:
        trans_list.append(T.Resize(crop,interpolation=3))
        pass
    return T.Compose(trans_list + [T.RandomHorizontalFlip(p=cfg.hf_p), *extra_t, base_transform()])


class RandomBlur(object):
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def __call__(self, image):
        r = random.uniform(self.r0, self.r1)
        return image.filter(ImageFilter.GaussianBlur(radius=r))
    pass


class MultiSample(object):
    """ generates n samples with augmentation """

    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))

    pass


class BaseDataset(metaclass=ABCMeta):

    def __init__(self, aug_cfg, bs_clf=1000, bs_test=1000):
        self.aug_cfg = aug_cfg
        self.bs_train, self.bs_clf, self.bs_test = self.aug_cfg.bs, bs_clf, bs_test
        self.num_workers = self.aug_cfg.num_workers
        pass

    @abstractmethod
    def ds_train(self):
        raise NotImplementedError

    @abstractmethod
    def ds_clf(self):
        raise NotImplementedError

    @abstractmethod
    def ds_clf_aug(self):
        raise NotImplementedError

    @abstractmethod
    def ds_test(self):
        raise NotImplementedError

    @property
    @lru_cache()
    def train(self):
        return DataLoader(dataset=self.ds_train(), batch_size=self.bs_train, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    @property
    @lru_cache()
    def clf(self):
        return DataLoader(dataset=self.ds_clf(), batch_size=self.bs_clf, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    @property
    @lru_cache()
    def clf_aug(self):
        return DataLoader(dataset=self.ds_clf_aug(), batch_size=self.bs_clf, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    @property
    @lru_cache()
    def test(self):
        return DataLoader(dataset=self.ds_test(), batch_size=self.bs_test, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=False)

    pass


class CIFAR10(BaseDataset):

    @staticmethod
    def base_transform():
        return T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    def ds_train(self):
        t = MultiSample(aug_transform(32, self.base_transform, self.aug_cfg), n=self.aug_cfg.num_samples)
        return C10(root=self.get_data_root(), train=True, download=True, transform=t)

    def ds_clf(self):
        t = self.base_transform()
        return C10(root=self.get_data_root(), train=True, download=True, transform=t)

    def ds_clf_aug(self):
        t = T.Compose([T.RandomCrop(32, padding=8), T.RandomHorizontalFlip(p=0.5),
                       T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        return C10(root=self.get_data_root(), train=True, download=True, transform=t)

    def ds_test(self):
        t = self.base_transform()
        return C10(root=self.get_data_root(), train=False, download=True, transform=t)

    @staticmethod
    def get_data_root(data_name="cifar10"):
        if data_name == "cifar10":
            data_root = ["/media/ubuntu/4T/ALISURE/Data/cifar",
                         "/mnt/4T/Data/data/CIFAR",
                         "/home/z840/private/ALISURE/Data/cifar"]
            for one in data_root:
                if os.path.exists(one):
                    return one
            pass
        return None

    pass


class CIFAR100(BaseDataset):

    @staticmethod
    def base_transform():
        return T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    @staticmethod
    def get_data_root(data_name="cifar100"):
        if data_name == "cifar100":
            data_root = "/media/ubuntu/4T/ALISURE/Data/cifar"
            if os.path.exists(data_root):
                return data_root
            else:
                data_root = "/mnt/4T/Data/data/CIFAR"
                if os.path.exists(data_root):
                    return data_root
            pass
        return None

    def ds_train(self):
        t = MultiSample(aug_transform(32, self.base_transform, self.aug_cfg), n=self.aug_cfg.num_samples)
        return C100(root=self.get_data_root(), train=True, download=True, transform=t,)

    def ds_clf(self):
        t = self.base_transform()
        return C100(root=self.get_data_root(), train=True, download=True, transform=t)

    def ds_clf_aug(self):
        t = self.base_transform()
        return C100(root=self.get_data_root(), train=True, download=True, transform=t)

    def ds_test(self):
        t = self.base_transform()
        return C100(root=self.get_data_root(), train=False, download=True, transform=t)
    pass


class STL10(BaseDataset):

    @staticmethod
    def base_transform():
        return T.Compose([T.ToTensor(), T.Normalize((0.43, 0.42, 0.39), (0.27, 0.26, 0.27))])

    @classmethod
    def test_transform(cls):
        return T.Compose([T.Resize(70, interpolation=3), T.CenterCrop(64), cls.base_transform()])

    @staticmethod
    def get_data_root(data_name="stl10"):
        if data_name == "stl10":
            # data_root = "/media/ubuntu/4T/ALISURE/Data/STL10"
            data_root = "/media/ubuntu/ALISURE/data/STL10"
            if os.path.exists(data_root):
                return data_root
            else:
                data_root = "/mnt/4T/Data/data/STL10"
                if os.path.exists(data_root):
                    return data_root
                else:
                    data_root = "/media/ubuntu/ALISURE-SSD/data/STL10"
                    if os.path.exists(data_root):
                        return data_root
                    pass
                pass
            pass
        return None

    def ds_train(self):
        t = MultiSample(aug_transform(64, self.base_transform, self.aug_cfg), n=self.aug_cfg.num_samples)
        return S10(root=self.get_data_root(), split="train+unlabeled", download=True, transform=t)

    def ds_clf(self):
        t = self.test_transform()
        return S10(root=self.get_data_root(), split="train", download=True, transform=t)

    def ds_clf_aug(self):
        t = self.base_transform()
        return S10(root=self.get_data_root(), split="train", download=True, transform=t)

    def ds_test(self):
        t = self.test_transform()
        return S10(root=self.get_data_root(), split="test", download=True, transform=t)
    pass


class TinyImageNet(BaseDataset):

    @staticmethod
    def base_transform():
        return T.Compose([T.ToTensor(), T.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))])

    @staticmethod
    def get_data_root(data_name="tiny-imagenet-200"):
        if data_name == "tiny-imagenet-200":
            data_root = "/media/ubuntu/4T/ALISURE/Data/tiny-imagenet-200"
            if os.path.exists(data_root):
                return data_root
            else:
                data_root = "/mnt/4T/Data/data/tiny-imagenet-200"
                if os.path.exists(data_root):
                    return data_root
                else:
                    data_root = "/media/ubuntu/4T/ALISURE/Data/Tiny-Imagenet-200"
                    if os.path.exists(data_root):
                        return data_root
                    pass
                pass
            pass
        return None

    def ds_train(self):
        t = MultiSample(aug_transform(64, self.base_transform, self.aug_cfg), n=self.aug_cfg.num_samples)
        return ImageFolder(root=os.path.join(self.get_data_root(), "tiny-imagenet-200/train"), transform=t)

    def ds_clf(self):
        t = self.base_transform()
        return ImageFolder(root=os.path.join(self.get_data_root(), "tiny-imagenet-200/train"), transform=t)

    def ds_clf_aug(self):
        t = self.base_transform()
        return ImageFolder(root=os.path.join(self.get_data_root(), "tiny-imagenet-200/train"), transform=t)

    def ds_test(self):
        t = self.base_transform()
        # return ImageFolder(root=os.path.join(self.get_data_root(), "tiny-imagenet-200/test"), transform=t)
        return ImageFolder(root=os.path.join(self.get_data_root(), "tiny-imagenet-200/val_new"), transform=t)
    pass


class ImageNet100(BaseDataset):

    @staticmethod
    def base_transform():
        return T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @staticmethod
    def test_transform():
        return T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @staticmethod
    def get_data_root(data_name="imagenet100"):
        if data_name == "imagenet100":
            data_root = [
                # "/media/ubuntu/ALISURE/data/ImageNet100",
                         "/media/ubuntu/4T/alisure_tmp/ImageNet/ImageNet100",
                         "/media/ubuntu/4T/ALISURE/Data/ImageNet/ImageNet100",
                         "/mnt/4T/Data/data/ImageNet/ImageNet100",
                         "/media/ubuntu/4T/ALISURE/Data/ImageNet/ImageNet100",
                         "/home/z840/private/ALISURE/Data/ImageNet100",
                         "/data/alisure/data/ImageNet100"]
            for one in data_root:
                if os.path.exists(one):
                    return one
            pass
        return None

    def ds_train(self):
        aug_with_blur = aug_transform(224, self.base_transform, self.aug_cfg,
                                      extra_t=[T.RandomApply([RandomBlur(0.1, 2.0)], p=0.5)])
        t = MultiSample(aug_with_blur, n=self.aug_cfg.num_samples)
        return ImageFolder(root=os.path.join(self.get_data_root(), "train"), transform=t)

    def ds_clf(self):
        t = self.test_transform()
        return ImageFolder(root=os.path.join(self.get_data_root(), "train"), transform=t)

    def ds_clf_aug(self):
        t = self.test_transform()
        return ImageFolder(root=os.path.join(self.get_data_root(), "train"), transform=t)

    def ds_test(self):
        t = self.test_transform()
        return ImageFolder(root=os.path.join(self.get_data_root(), "val"), transform=t)

    pass

