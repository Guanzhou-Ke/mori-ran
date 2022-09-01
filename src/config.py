"""
Configuration loading tools.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import os

from yacs.config import CfgNode as CN


_C = CN()

# enable gpu
_C.gpu = True
# gpu device id.
_C.device = 0
# task type ['clustering', 'classification']
_C.task = 'clustering'
# test time defaults to 5
_C.test_time = 5
# seed
_C.seed = 42
# print log
_C.verbose = True
# experiment name
_C.experiment = ''
# the interval of record epoch, defaults to 10.
_C.record = 10
# For multi-view setting
_C.views = None
_C.hidden_dim = 128
_C.experiment_id = 0


# Network setting.
_C.backbone = CN()
_C.backbone.type = None
_C.backbone.arch0 = None
_C.backbone.arch1 = None

# for training.
_C.train = CN()
_C.train.epochs = 100
# ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST']
_C.train.dataset = 'Scene-15'
_C.train.batch_size = None
_C.train.optim = None
_C.train.lr = 0.001
_C.train.num_workers = 8
_C.train.save_log = True
# if None, it will be set as './experiments/results/[model name]/[dataset name]'
_C.train.log_dir = None
# mix precision.
_C.train.fp16 = True
_C.train.opt_level = 'O1'

# For fusion
_C.fusion = CN()
_C.fusion.activation = 'relu'
_C.fusion.hidden_dim = 128
_C.fusion.use_bn = True
_C.fusion.mlp_layers = 2
_C.fusion.save_embeddings = -1


# For Conan
_C.cluster_module = CN()
_C.cluster_module.type = 'ddc'
_C.cluster_module.num_cluster = None
_C.cluster_module.hidden_dim = None # ddc input features
_C.cluster_module.cluster_hidden_dim = None # ddc hidden features

_C.contrastive = CN()
_C.contrastive.ins_enable = True
_C.contrastive.cls_enable = True
_C.contrastive.type = 'simclr'
_C.contrastive.projection_dim = None
_C.contrastive.ins_lambda = 0.5
_C.contrastive.cls_lambda = 0.5
_C.contrastive.con_lambda = 0.01
_C.contrastive.projection_layers = 2
_C.contrastive.temperature = 0.07
_C.contrastive.nmb_protos = 256
_C.contrastive.eps = 0.05
_C.contrastive.ds_iters = 3
_C.contrastive.symmetry = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project.

    Returns:
        CfgNode: configuration.
    """
    return _C.clone()


def get_cfg(args):
    config = get_cfg_defaults()
    config.experiment = args.experiment
    config.train.dataset = args.dataset
    if args.ablation:
        config.merge_from_file(os.path.join('./experiments/configs', f"{args.experiment}/{args.experiment}-{args.dataset}-ablation.yaml"))
    else:
        config.merge_from_file(os.path.join('./experiments/configs', f"{args.experiment}/{args.experiment}-{args.dataset}.yaml"))
    if not config.train.log_dir:
        if args.ablation:
            path = f'./experiments/results/ablation/{args.experiment}-{config.experiment_id}/{args.dataset}/{config.task}'
        else:
            path = f'./experiments/results/{args.experiment}-{config.experiment_id}/{args.dataset}/{config.task}'
        os.makedirs(path, exist_ok=True)
        config.train.log_dir = path
    config.freeze()
    return config



