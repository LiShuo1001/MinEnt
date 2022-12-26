import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from functools import partial
from torchvision import models
import torch.nn.functional as F

from const import NormName


def eval_sgd(x_train, y_train, x_test, y_test, topk=[1, 5], epoch=500, has_tqdm=False, bs=1000):
    lr_start, lr_end = 1e-2, 1e-6
    gamma = (lr_end / lr_start) ** (1 / epoch)
    output_size = x_train.shape[1]
    num_class = y_train.max().item() + 1
    clf = nn.Linear(output_size, num_class)
    clf.cuda()
    clf.train()
    optimizer = optim.Adam(clf.parameters(), lr=lr_start, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    range_epoch = tqdm(range(epoch), total=epoch) if has_tqdm else range(epoch)
    for ep in range_epoch:
        perm = torch.randperm(len(x_train)).view(-1, bs)
        for idx in perm:
            optimizer.zero_grad()
            criterion(clf(x_train[idx]), y_train[idx]).backward()
            optimizer.step()
        scheduler.step()
        pass

    clf.eval()
    with torch.no_grad():
        y_pred = clf(x_test)
    pred_top = y_pred.topk(max(topk), 1, largest=True, sorted=True).indices
    acc = {t: (pred_top[:, :t] == y_test[..., None]).float().sum(1).mean().cpu().item() for t in topk}
    del clf
    return acc


def eval_knn(x_train, y_train, x_test, y_test, k=5):
    """ k-nearest neighbors classifier accuracy """
    d = torch.cdist(x_test, x_train)
    topk = torch.topk(d, k=k, dim=1, largest=False)
    labels = y_train[topk.indices]
    pred = torch.empty_like(y_test)
    for i in range(len(labels)):
        x = labels[i].unique(return_counts=True)
        pred[i] = x[0][x[1].argmax()]

    acc = (pred == y_test).float().mean().cpu().item()
    del d, topk, labels, pred
    return acc


def get_data(model, loader, output_size):
    xs = torch.empty(len(loader), loader.batch_size, output_size, dtype=torch.float32).cuda()
    ys = torch.empty(len(loader), loader.batch_size, dtype=torch.long).cuda()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
            x = x.cuda()
            xs[i] = model(x).cuda()
            ys[i] = y.cuda()
    xs = xs.view(-1, output_size)
    ys = ys.view(-1)
    return xs, ys


####################################################################################################################


class NormMethod(nn.Module):

    def __init__(self, emb, norm_name=NormName.bn):
        super().__init__()
        self.norm_name = norm_name

        if self.norm_name == NormName.bn:
            self.norm_layer = nn.BatchNorm1d(emb, affine=True)
        elif self.norm_name == NormName.softmax:
            self.norm_layer = nn.Softmax(dim=0)
        elif self.norm_name == NormName.l2norm:
            self.norm_layer = partial(nn.functional.normalize, p=2, dim=0)
        elif self.norm_name == NormName.no:
            self.norm_layer = nn.Identity()
        else:
            self.norm_layer = nn.BatchNorm1d(emb, affine=True)
        pass

    def forward(self, x):
        return self.norm_layer(x)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    pass


class BaseMethod(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model, self.out_size = self.get_backbone(cfg)
        pass

    @staticmethod
    def get_backbone(cfg):
        model = getattr(models, cfg.arch)(pretrained=False)
        if "imagenet" not in cfg.dataset_name:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if cfg.dataset_name == "cifar10" or cfg.dataset_name == "cifar100":
            model.maxpool = nn.Identity()
        out_size = model.fc.in_features
        model.fc = nn.Identity()
        return model, out_size

    @staticmethod
    def get_head(out_size, cfg, emb):
        x = []
        in_size = out_size
        for _ in range(cfg.head_layers - 1):
            x.append(nn.Linear(in_size, cfg.head_size))
            if cfg.add_bn:
                x.append(nn.BatchNorm1d(cfg.head_size))
            x.append(nn.ReLU())
            in_size = cfg.head_size
        x.append(nn.Linear(in_size, emb))
        return nn.Sequential(*x)

    def forward(self, samples):
        raise NotImplementedError

    def get_acc(self, ds_clf, ds_test, knn, sgd_epoch=500):
        self.eval()
        model, out_size = self.model, self.out_size

        x_train, y_train = get_data(model, ds_clf, out_size)
        x_test, y_test = get_data(model, ds_test, out_size)

        acc_knn = eval_knn(x_train, y_train, x_test, y_test, knn)
        acc_linear = eval_sgd(x_train, y_train, x_test, y_test, epoch=sgd_epoch, bs=ds_test.batch_size)
        del x_train, y_train, x_test, y_test
        self.train()
        return acc_knn, acc_linear

    def step(self, progress):
        pass

    pass


class MinEntOneHead(BaseMethod):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.head = self.get_head(self.out_size, cfg, cfg.emb)
        self.norm = NormMethod(cfg.emb, norm_name=cfg.norm_name)

        self.criterion = nn.CrossEntropyLoss()
        pass

    def forward(self, samples):
        f1 = self.model(samples[0].cuda(non_blocking=True))
        f2 = self.model(samples[1].cuda(non_blocking=True))
        g1, g2 = self.head(f1), self.head(f2)
        h1, h2 = self.norm(g1), self.norm(g2)
        t1, t2 = h1.argmax(dim=1), h2.argmax(dim=1)
        return 0.5 * (self.criterion(g1, t2) + self.criterion(g2, t1))

    pass


class MinEntMultiHead(BaseMethod):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.head_list = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        for emb in cfg.emb_list:
            self.head_list.append(self.get_head(self.out_size, cfg, emb))
            self.norm_list.append(NormMethod(emb, norm_name=cfg.norm_name))
            pass

        self.criterion = nn.CrossEntropyLoss()
        pass

    def forward(self, samples):
        f1 = self.model(samples[0].cuda(non_blocking=True))
        f2 = self.model(samples[1].cuda(non_blocking=True))
        loss = 0
        for head, norm in zip(self.head_list, self.norm_list):
            g1, g2 = head(f1), head(f2)
            h1, h2 = norm(g1), norm(g2)
            t1, t2 = h1.argmax(dim=1), h2.argmax(dim=1)
            loss += 0.5 * (self.criterion(g1, t2) + self.criterion(g2, t1))
            pass
        return loss / len(self.head_list)

    pass


####################################################################################################################

