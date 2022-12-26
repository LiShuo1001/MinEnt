import os
import torch
import numpy as np
import torch.optim as optim
from alisuretool.Tools import Tools
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts

from config import get_cfg
from model import MinEntMultiHead, MinEntOneHead
from const import CompareMethod, DatasetName, OptimName
from data import CIFAR10, CIFAR100, STL10, TinyImageNet, ImageNet100


def set_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(optimizer, T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
                                           T_mult=cfg.Tmult, eta_min=cfg.eta_min)
    elif cfg.lr_step == "step":
        return MultiStepLR(optimizer, milestones=[cfg.epoch - a for a in cfg.drop], gamma=cfg.drop_gamma)
    else:
        return None
    pass


def runner_main(cfg):
    cfg.name = "{}_{}_{}_{}".format(cfg.dataset_name, cfg.method_name, cfg.optim, cfg.norm_name)

    cfg.model.cuda().train()
    cudnn.benchmark = True

    Tools.print(cfg)
    Tools.print()
    Tools.print(f"optim={cfg.optim} lr_step={cfg.lr_step}")
    if cfg.optim == "adam":
        optimizer = optim.Adam(cfg.model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)
    else:
        optimizer = optim.SGD(cfg.model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = set_scheduler(optimizer, cfg)

    checkpoint_path = f"./checkpoint/ICML/{cfg.method_name }/{cfg.dataset_name}/{cfg.name}"
    log_file = Tools.new_dir(os.path.join(checkpoint_path, "log.txt"))

    acc_knn, acc = cfg.model.get_acc(cfg.dataset.clf, cfg.dataset.test, cfg.knn)
    Tools.print("name={} acc={:.4f} acc_5={:.4f} acc_knn={:.4f}".format(
        cfg.name, acc[1], acc[5], acc_knn), txt_path=log_file)

    lr_warmup = 0
    best_acc = 0.0
    eval_every = cfg.eval_every
    for ep in range(cfg.epoch):
        loss_ep, stat_ep = [], []

        for n_iter, (samples, _) in enumerate(cfg.dataset.train):
            if lr_warmup < 500:
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * (lr_warmup + 1) / 500
                lr_warmup += 1
                pass

            optimizer.zero_grad()

            if cfg.stat:
                loss, stat = cfg.model(samples)
                stat_ep.append(stat.item())
            else:
                loss = cfg.model(samples)

            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            cfg.model.step(ep / cfg.epoch)

            # cos
            if cfg.lr_step == "cos" and lr_warmup >= 500:
                scheduler.step(ep + n_iter / len(cfg.dataset.train))
                pass
            pass

        # step
        if cfg.lr_step == "step":
            scheduler.step()
            pass

        if len(cfg.drop) and ep == (cfg.epoch - cfg.drop[0]):
            eval_every = cfg.eval_every_drop
            pass

        if (ep + 1) % eval_every == 0:
            acc_knn, acc = cfg.model.get_acc(cfg.dataset.clf, cfg.dataset.test, cfg.knn)
            Tools.print("name={} acc={:.4f} acc_5={:.4f} acc_knn={:.4f}".format(
                cfg.name, acc[1], acc[5], acc_knn), txt_path=log_file)
            if acc[1] >= best_acc:
                best_acc = acc[1]
                new_file = Tools.new_dir(os.path.join(checkpoint_path, f"{ep}_{best_acc}.pt"))
                torch.save(cfg.model.state_dict(), new_file)
                Tools.print("save to {}".format(new_file), txt_path=log_file)
            pass

        if cfg.stat:
            Tools.print("name={} ep={} lr={:.8f} loss={:.4f} stat={:.4f}".format(
                cfg.name, ep, scheduler.get_last_lr()[0], np.mean(loss_ep), np.mean(stat_ep)), txt_path=log_file)
        else:
            Tools.print("name={} ep={} lr={:.8f} loss={:.4f}".format(
                cfg.name, ep, scheduler.get_last_lr()[0], np.mean(loss_ep)), txt_path=log_file)
        pass

    pass


def set_optim(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    if cfg.optim == OptimName.adam:
        cfg.optim = "adam"
        cfg.lr_step = "step"
        cfg.lr = 0.001
    elif cfg.optim == OptimName.sgd:
        cfg.optim = "sgd"
        cfg.lr_step = "cos"
        cfg.lr = 0.01
        pass
    return cfg


def set_method(cfg):
    if cfg.method_name == CompareMethod.MinEntOneHead:
        cfg.model = MinEntOneHead(cfg)
    elif cfg.method_name == CompareMethod.MinEntMultiHead:
        cfg.emb = None
        if cfg.emb_list is None:
            cfg.emb_list = [2048, 1024, 512, 256, 128, 64, 32, 16]
            if cfg.dataset_name == DatasetName.stl10 or cfg.dataset_name == DatasetName.tiny:
                cfg.emb_list = [8192, 4096, 2048, 1024, 512, 256, 128, 64]
        cfg.model = MinEntMultiHead(cfg)
    return cfg


def set_dataset(cfg):
    if cfg.dataset_name == DatasetName.cifar10:
        cfg.dataset = CIFAR10(cfg)
    elif cfg.dataset_name == DatasetName.cifar100:
        cfg.dataset = CIFAR100(cfg)
    elif cfg.dataset_name == DatasetName.stl10:
        cfg.epoch = 2000
        cfg.dataset = STL10(cfg)
    elif cfg.dataset_name == DatasetName.tiny:
        cfg.dataset = TinyImageNet(cfg)
    elif cfg.dataset_name == DatasetName.imagenet100:
        cfg.num_workers = 32
        cfg.epoch = 500
        cfg.crop_s0 = 0.08
        cfg.cj0 = 0.8
        cfg.cj1 = 0.8
        cfg.cj2 = 0.8
        cfg.cj3 = 0.2
        cfg.gs_p = 0.2
        cfg.dataset = ImageNet100(cfg)
        pass
    return cfg


if __name__ == "__main__":
    cfg = get_cfg()
    cfg = set_optim(cfg=cfg)
    cfg = set_method(cfg=cfg)
    cfg = set_dataset(cfg=cfg)

    runner_main(cfg=cfg)
    pass
