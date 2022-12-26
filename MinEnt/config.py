import argparse
import multiprocessing
from functools import partial
from torchvision import models
from const import DatasetName, CompareMethod, NormName, OptimName


def get_cfg():
    parser = argparse.ArgumentParser(description="")

    ##################################################################################################################
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--dataset_name", type=str, choices=[
        DatasetName.cifar10, DatasetName.cifar100, DatasetName.stl10, DatasetName.tiny, DatasetName.imagenet100],
                        default=DatasetName.cifar10, help="dataset name")
    parser.add_argument("--method_name", type=str, choices=[
        CompareMethod.MinEntMultiHead, CompareMethod.MinEntOneHead],
                        default=CompareMethod.MinEntMultiHead, help="method name")
    parser.add_argument("--norm_name", type=str, default=NormName.bn, choices=[
        NormName.bn, NormName.softmax, NormName.l2norm, NormName.no], help="norm name")
    parser.add_argument("--optim", type=str, default=OptimName.adam, choices=[
        OptimName.adam, OptimName.sgd], help="optim name")
    ##################################################################################################################

    addf = partial(parser.add_argument, type=float)
    addf("--cj0", default=0.4, help="color jitter brightness")
    addf("--cj1", default=0.4, help="color jitter contrast")
    addf("--cj2", default=0.4, help="color jitter saturation")
    addf("--cj3", default=0.1, help="color jitter hue")
    addf("--cj_p", default=0.8, help="color jitter probability")
    addf("--gs_p", default=0.1, help="grayscale probability")
    addf("--crop_s0", default=0.2, help="crop size from")
    addf("--crop_s1", default=1.0, help="crop size to")
    addf("--crop_r0", default=0.75, help="crop ratio from")
    addf("--crop_r1", default=(4 / 3), help="crop ratio to")
    addf("--hf_p", default=0.5, help="horizontal flip probability")

    parser.add_argument("--bs", type=int, default=128, help="total epoch number")
    parser.add_argument("--epoch", type=int, default=1000, help="total epoch number")
    parser.add_argument("--head_layers", type=int, default=3, help="number of FC layers in head")
    parser.add_argument("--head_size", type=int, default=1024, help="size of FC layers in head")
    parser.add_argument("--emb", type=int, default=128, help="size of output layer in head")
    parser.add_argument("--m", type=int, default=8, help="number of multi-head")

    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count() // 4, help="dataset workers number")
    parser.add_argument("--lr_step", type=str, choices=["cos", "step", "none"], default="step", help="learning rate schedule type")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--adam_l2", type=float, default=1e-6, help="weight decay (L2 penalty)")
    parser.add_argument("--eta_min", type=float, default=0, help="min learning rate (for --lr_step cos)")
    parser.add_argument("--T0", type=int, help="period (for --lr_step cos)")
    parser.add_argument("--Tmult", type=int, default=1, help="period factor (for --lr_step cos)")
    parser.add_argument("--num_samples", type=int, default=2, help="number of samples (d) generated from each image")

    parser.add_argument("--knn", type=int, default=5, help="k in k-nn classifier")
    parser.add_argument("--eval_every_drop", type=int, default=5, help="how often to evaluate after learning rate drop")
    parser.add_argument("--eval_every", type=int, default=10, help="how often to evaluate")
    parser.add_argument("--drop", type=int, nargs="*", default=[50, 25], help="milestones for learning rate decay (0 = last epoch)")
    parser.add_argument("--drop_gamma", type=float, default=0.2, help="multiplicative factor of learning rate decay")

    parser.add_argument("--arch", type=str, choices=[x for x in dir(models) if "resn" in x], default="resnet18", help="encoder architecture")
    parser.add_argument("--clf", type=str, default="sgd", choices=["sgd", "knn", "lbfgs"], help="classifier for test.py")

    parser.add_argument("--no_stat", dest="stat", action="store_true", help="don't normalize latents",)
    parser.add_argument("--no_add_bn", dest="add_bn", action="store_false", help="do not use BN in head")

    my_args = parser.parse_args()

    my_args.stat = False
    my_args.has_color_jitter = True
    my_args.has_gray_scale = True
    my_args.has_crop = True
    my_args.emb_list = None
    return my_args
