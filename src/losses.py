import torch
from torch import nn


class BarlowTwinsModule(nn.Module):
    """
    References to: FAIR's barlowtwins.
    """
    
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        # according to original paper 2048 -> 8192
        self.projector_dim = args.hidden_dim * 4
        self.symmetry = args.contrastive.symmetry
        layers = []
        layers.append(nn.Linear(args.hidden_dim, self.projector_dim, bias=False))
        layers.append(nn.BatchNorm1d(self.projector_dim))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(args.contrastive.projection_layers - 2):
            layers.append(nn.Linear(self.projector_dim, self.projector_dim, bias=False))
            layers.append(nn.BatchNorm1d(self.projector_dim))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Linear(self.projector_dim, self.projector_dim, bias=False))
        self.projector = nn.Sequential(*layers)
        
        self.bn = nn.BatchNorm1d(self.projector_dim, affine=False)
        
        
        
    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
        
    def get_loss(self, z, hs):
        z_p = self(z)
        loss = 0
        for h in hs:
            h_p = self(h)
            
            c = self.bn(z_p).T @ self.bn(h_p)
            
            c.div_(self.args.train.batch_size)
            
            on_diag = torch.diagonal(c).add(-1).pow(2).sum()
            off_diag = self._off_diagonal(c).pow(2).sum()
            loss += on_diag + self.args.contrastive.con_lambda * off_diag
        
        if self.symmetry:
            # just for ablation study.
            h1_p = self(hs[0])
            h2_p = self(hs[1])
            
            c = self.bn(h1_p).T @ self.bn(h2_p)
            
            c.div_(self.args.train.batch_size)
            
            on_diag = torch.diagonal(c).add(-1).pow(2).sum()
            off_diag = self._off_diagonal(c).pow(2).sum()
            loss += on_diag + self.args.contrastive.con_lambda * off_diag
            
            
        return loss
    
    def forward(self, x):
        return self.projector(x)
    
    
class CategoryContrastiveModule(nn.Module):
    """
    References to: SWAV.
    """
    
    def __init__(self, args, device) -> None:
        super().__init__()
        self.args = args
        self.input_dim = args.hidden_dim # 128
        self.projector_dim = args.contrastive.projection_dim # 2048
        self.nmb_protos = args.contrastive.nmb_protos # 256
        self.eps = args.contrastive.eps
        self.device = device
        self.ds_iters = args.contrastive.ds_iters
        self.temperature = args.contrastive.temperature
        self.projection_head = nn.Sequential(
            nn.Linear(self.input_dim , self.projector_dim),
            nn.BatchNorm1d(self.projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projector_dim, self.input_dim)
        )
        
        self.prototypes = nn.Linear(self.input_dim, self.nmb_protos, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)
    
    
    def get_loss(self, z, hs):
        
        # normalize
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
            
        z_p = self(z)
        qz = torch.exp(z_p / self.eps)
        qz = self.distributed_sinkhorn(qz, self.ds_iters, self.device)
        loss = 0
        for h in hs:
            h_p = self(h)
            ph = self.softmax(h_p / self.temperature)
            loss -= torch.mean(torch.sum(qz @ torch.log(ph), dim=1))
            
        qh = self(hs[0])
        ph = self(hs[1])
        qh = torch.exp(qh / self.eps)
        qh = self.distributed_sinkhorn(qh, self.ds_iters, self.device)
        ph = self.softmax(ph / self.temperature)
        
        loss -= torch.mean(torch.sum(qh @ torch.log(ph), dim=1))
        
        return loss
            
        
    def distributed_sinkhorn(self, Q, nmb_iters, device):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0]).to(device, non_blocking=True)
            r = torch.ones(Q.shape[0]).to(device, non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).to(device, non_blocking=True) / Q.shape[1]

            curr_sum = torch.sum(Q, dim=1)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
                
        
        
        