"""
Model.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import math

import torch
from torch import nn
from torch.nn import init

from networks import build_mlp
from losses import BarlowTwinsModule, CategoryContrastiveModule


class MORIRAN(nn.Module):
    """MORIRAN
    """
    
    def __init__(self, args, device='cpu') -> None:
        super().__init__()
        
        self.args = args
        self.device = device
        
        self.encoder0 = build_mlp(args.backbone.arch0)
        self.encoder1 = build_mlp(args.backbone.arch1)

        self.fusion_layer = FusionBlock(args)
        
        if self.args.contrastive.ins_enable:
            self.instance_loss = BarlowTwinsModule(args)
        if self.args.contrastive.cls_enable:
            self.cls_loss = CategoryContrastiveModule(args, device)
            
        self.classification = False  
        if self.args.task == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(self.args.hidden_dim, self.args.cluster_module.num_cluster)
            )
            self.supervised_loss = nn.CrossEntropyLoss()
            self.classification = True
        
        self.apply(self.weights_init('xavier'))
        
        
    def _get_hs(self, Xs):
        hs = [bb(x) for bb, x in zip([self.__getattr__(f"encoder{idx}") for idx in range(self.args.views)], Xs)]
        return hs

    def forward(self, Xs):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        if self.classification:
            self.classifier(z)
        return z
    
    @torch.no_grad()
    def predict(self, Xs):
        y = self(Xs)
        _, predicted = torch.max(y, 1)
        return predicted
    
    @torch.no_grad()
    def commonZ(self, Xs):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        return z

    @torch.no_grad()
    def extract_all_hidden(self, Xs):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        return hs, z
    
    def get_loss(self, Xs, y=None):
        hs = self._get_hs(Xs)
        z = self.fusion_layer(hs)
        
        if y is not None and self.classification:
            z = self.classifier(z)
            loss = self.supervised_loss(z, y)
            return loss
        else:
            if self.args.contrastive.ins_enable:
                loss_ins = self.instance_loss.get_loss(z, hs)
            else:
                loss_ins = 0
            if self.args.contrastive.cls_enable:
                loss_cls = self.cls_loss.get_loss(z, hs)
            else:
                loss_cls = 0
            
            total_loss = self.args.contrastive.ins_lambda * loss_ins + self.args.contrastive.cls_lambda * loss_cls
            
            return total_loss, (loss_ins, loss_cls)
    
    def weights_init(self, init_type='gaussian'):
        def init_fun(m):
            classname = m.__class__.__name__
            if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
                # print m.__class__.__name__
                if init_type == 'gaussian':
                    init.normal_(m.weight, 0.0, 0.02)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight, gain=math.sqrt(2))
                elif init_type == 'default':
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0.0)

        return init_fun
        
            
        
class FusionBlock(nn.Module):
    """Fusion block with residual connection.
    
       A block like to:
        input -> (_, 512) -> (_, 256) -> (_, 512) -> output
          |                                   |
          ------------------+------------------
    """

    def __init__(self, args):
        super(FusionBlock, self).__init__()
        act_func = args.fusion.activation
        views = args.views
        use_bn = args.fusion.use_bn
        mlp_layers = args.fusion.mlp_layers
        in_features = args.hidden_dim
        latent_dim = in_features // 2
        if act_func == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        
        self.map_layer = nn.Sequential(
            nn.Linear(in_features*views, in_features, bias=True),
            nn.BatchNorm1d(in_features),
            self.act,
            )
        self.fusion = nn.Sequential(
            nn.Linear(in_features, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            self.act,
            self._make_layers(latent_dim, latent_dim, self.act, mlp_layers, use_bn),
            nn.Linear(latent_dim, in_features, bias=False),
        )
        

    def forward(self, h):
        h = torch.cat(h, dim=-1)
        z = self.map_layer(h)
        res = z
        z = self.fusion(z)
        z += res
        return z

    def _make_layers(self, in_features, out_features, act, num_layers, bn=False):
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, out_features, bias=False))
            if bn:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(act)
        return nn.Sequential(*layers)
    
    
if __name__ == '__main__':
    import os
    from config import get_cfg_defaults
    from utils import print_network
    config = get_cfg_defaults()
    config.experiment = 'moriran'
    config.train.dataset = 'LandUse-21'
    config.merge_from_file(os.path.join('./experiments/configs', f"{config.experiment}-{config.train.dataset }.yaml"))
    print(config.fusion.mlp_layers)
    x1, x2 = torch.rand(100, 59), torch.rand(100, 40)
    moriran = MORIRAN(config)
    print_network(moriran)
    
    z = moriran([x1, x2])
    print(z.shape)   
    
    loss, _ = moriran.get_loss([x1, x2])
    print(loss)