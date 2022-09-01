import math
from typing import Optional
from abc import ABC

import torch
from torch import nn
from torch.nn import init
from torch.nn import Parameter
import numpy as np
from torch.nn import functional as F

from networks import build_mlp


class CONAN(nn.Module):
    def __init__(self, args, device='cpu'):
        super(CONAN, self).__init__()
        self.args = args
        self.device = device
        
        self.encoder0 = build_mlp(args.backbone.arch0)
        self.encoder1 = build_mlp(args.backbone.arch1)

        self.fusion_layer = FusionLayer(args)

        self.clustering_module, self.cls_criterion = build_clustering_module(args, 
                                                                             self.device)

        self.contrastive_module = build_contrastive_module(args, self.device)
            
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
            y = self.classifier(z)
            return y
        if self.args.cluster_module.type == 'dec':
            y = self.clustering_module(z)
        elif self.args.cluster_module.type == 'ddc':
            y, _ = self.clustering_module(z)
        else:
            raise ValueError('Clustering loss error.')
        return y

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
        if self.args.cluster_module.type == 'dec':
            y = self.clustering_module(z)
            clustering_loss = self.cls_criterion(y)
        elif self.args.cluster_module.type == 'ddc':
            y, h = self.clustering_module(z)
            clustering_loss = self.cls_criterion(y, h)
        else:
            raise ValueError('Clustering loss error.')
        contrastive_loss = self.contrastive_module.get_loss(z, hs)
        tot_loss = clustering_loss + contrastive_loss
        return tot_loss, (clustering_loss, contrastive_loss)

    @torch.no_grad()
    def predict(self, Xs):
        return self(Xs).detach().cpu().max(1)[1]

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

def build_clustering_module(args, device):
    if args.cluster_module.type == 'dec':
        clustering_layer = DECModule(cluster_number=args.cluster_module.num_cluster,
                                                  embedding_dimension=args.cluster_module.hidden_dim)
        cls_criterion = DECLoss()
    elif args.cluster_module.type == 'ddc':
        clustering_layer = DDCModule(args.cluster_module.hidden_dim, 
                                     args.cluster_module.cluster_hidden_dim, 
                                     args.cluster_module.num_cluster)
        cls_criterion = DDCLoss(args.cluster_module.num_cluster, device=device)
    else:
        raise ValueError('Loss type must be dec or ddc.')
    return clustering_layer, cls_criterion


def build_contrastive_module(args, device):
    if args.contrastive.type == 'simsiam':
        return SimSiamModule(args, device)
    elif args.contrastive.type == 'simclr':
        return SimCLRModule(args, device)
    else:
        raise ValueError('Contrastive type error.')


class FusionLayer(nn.Module):

    def __init__(self, args):
        super(FusionLayer, self).__init__()
        act_func = args.fusion.activation
        views = args.views
        use_bn = args.fusion.use_bn
        mlp_layers = args.fusion.mlp_layers
        in_features = args.fusion.hidden_dim
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'tanh':
            self.act = nn.Tanh()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError('Activate function must be ReLU or Tanh.')
        self.layers = [self._make_layers(in_features * views, in_features, self.act, use_bn)]
        if mlp_layers > 1:
            layers = [self._make_layers(in_features, in_features,
                                        self.act if _ < (mlp_layers - 2) else nn.Identity(),
                                        use_bn if _ < (mlp_layers - 2) else False) for _ in range(mlp_layers - 1)]
            self.layers += layers
        self.layers = nn.Sequential(*self.layers)

    def forward(self, h):
        h = torch.cat(h, dim=-1)
        z = self.layers(h)
        return z

    def _make_layers(self, in_features, out_features, act, bn=False):
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features, out_features))
        layers.append(act)
        if bn:
            layers.append(nn.BatchNorm1d(out_features))
        return nn.Sequential(*layers)


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048, num_layers=2):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        if self.num_layers == 3:
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiamModule(nn.Module):

    def __init__(self, args, device):
        super(SimSiamModule, self).__init__()
        in_features = args.hidden_dim
        self.projection = projection_MLP(in_features,
                                         hidden_dim=args.projection_dim,
                                         out_dim=args.cluster_hidden_dim,
                                         num_layers=args.projection_layers)
        self.predictor = prediction_MLP(args.cluster_hidden_dim,
                                        hidden_dim=args.prediction_hidden_dim,
                                        out_dim=args.cluster_hidden_dim)
        self.con_criterion = SimSiamLoss()
        self.con_lambda = args.contrastive_lambda


    def forward(self, h):
        p = self.projection(h)
        z = self.predictor(p)
        return p, z

    def get_loss(self, ch, hs):
        cp, cz = self(ch)
        sub_loss = 0
        for h in hs:
            p, z = self(h)
            sub_loss += self.con_criterion(cp, p, cz, z)
        return self.con_lambda * sub_loss


class SimCLRModule(nn.Module):

    def __init__(self, args, device='cpu'):
        super(SimCLRModule, self).__init__()
        in_features = args.hidden_dim
        if args.contrastive.projection_layers == 0:
            self.projection = nn.Identity()
        else:
            self.projection = projection_MLP(in_features,
                                             hidden_dim=args.contrastive.projection_dim,
                                             out_dim=args.cluster_module.cluster_hidden_dim,
                                             num_layers=args.contrastive.projection_layers)
        self.con_criterion = SimCLRLoss(args)
        self.con_lambda = args.contrastive.con_lambda
        self.args = args

    def forward(self, h):
        h = self.projection(h)
        return h

    def get_loss(self, ch, hs):
        cp = self(ch)
        sub_loss = 0
        for h in hs:
            p = self(h)
            ps = torch.cat([cp, p], dim=-1)
            sub_loss += self.con_criterion(ps)
        return self.con_lambda * sub_loss


class DECModule(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(DECModule, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class DDCModule(nn.Module):

    def __init__(self, in_features, hidden_dim, num_cluster):
        super(DDCModule, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim, momentum=0.1)
        )

        self.clustering_layer = nn.Sequential(
            nn.Linear(hidden_dim, num_cluster),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        y = self.clustering_layer(h)
        return y, h


class BaseLoss:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class DECLoss(BaseLoss, ABC):
    """
    Deep embedding clustering.
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis."
    International conference on machine learning. PMLR, 2016.
    """

    def __init__(self):
        super(DECLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')

    def __call__(self, logist, **kwargs):
        Q = self.target_distribution(logist).detach()
        loss = self.criterion(logist.log(), Q) / logist.shape[0]
        return loss

    def target_distribution(self, logist) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (logist ** 2) / torch.sum(logist, 0)
        return (weight.t() / torch.sum(weight, 1)).t()


class DDCLoss(BaseLoss, ABC):
    """
    Michael Kampffmeyer et al. "Deep divergence-based approach to clustering"
    """

    def __init__(self, num_cluster, epsilon=1e-9, rel_sigma=0.15, device='cpu'):
        """

        :param epsilon:
        :param rel_sigma: Gaussian kernel bandwidth
        """
        super(DDCLoss, self).__init__()
        self.epsilon = epsilon
        self.rel_sigma = rel_sigma
        self.device = device
        self.num_cluster = num_cluster

    def __call__(self, logist, hidden):
        hidden_kernel = self._calc_hidden_kernel(hidden)
        l1_loss = self._l1_loss(logist, hidden_kernel, self.num_cluster)
        l2_loss = self._l2_loss(logist)
        l3_loss = self._l3_loss(logist, hidden_kernel, self.num_cluster)
        return l1_loss + l2_loss + l3_loss

    def _l1_loss(self, logist, hidden_kernel, num_cluster):
        return self._d_cs(logist, hidden_kernel, num_cluster)

    def _l2_loss(self, logist):
        n = logist.size(0)
        return 2 / (n * (n - 1)) * self._triu(logist @ torch.t(logist))

    def _l3_loss(self, logist, hidden_kernel, num_cluster):
        if not hasattr(self, 'eye'):
            self.eye = torch.eye(num_cluster, device=self.device)
        m = torch.exp(-self._cdist(logist, self.eye))
        return self._d_cs(m, hidden_kernel, num_cluster)

    def _triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _calc_hidden_kernel(self, x):
        return self._kernel_from_distance_matrix(self._cdist(x, x), self.epsilon)

    def _d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.

        :param A: Cluster assignment matrix
        :type A:  torch.Tensor
        :param K: Kernel matrix
        :type K: torch.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: torch.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom, eps=self.epsilon)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=self.epsilon ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self._triu(nom / torch.sqrt(dnom_squared))
        return d

    def _atleast_epsilon(self, X, eps):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: torch.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: torch.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def _cdist(self, X, Y):
        """
        Pairwise distance between rows of X and rows of Y.

        :param X: First input matrix
        :type X: torch.Tensor
        :param Y: Second input matrix
        :type Y: torch.Tensor
        :return: Matrix containing pairwise distances between rows of X and rows of Y
        :rtype: torch.Tensor
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    def _kernel_from_distance_matrix(self, dist, min_sigma):
        """
        Compute a Gaussian kernel matrix from a distance matrix.

        :param dist: Disatance matrix
        :type dist: torch.Tensor
        :param min_sigma: Minimum value for sigma. For numerical stability.
        :type min_sigma: float
        :return: Kernel matrix
        :rtype: torch.Tensor
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = self.rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(- dist / (2 * sigma2))
        return k


class SimSiamLoss(BaseLoss, ABC):
    """
    SimSiam Loss.
    Negative cosine similarity.

    """

    def __call__(self, p1, p2, z1, z2):
        return self._D(p1, z2) / 2  + self._D(p2, z1) / 2

    def _D(self, p, z):
        """
        The original implementation like below, but we could try the faster version.
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize
        z = F.normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()
        :param p:
        :param z:
        :return:
        """
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()


class SimCLRLoss(BaseLoss, ABC):
    large_num = 1e9

    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.temp = args.contrastive.temperature

    @staticmethod
    def _norm(mat):
        return torch.nn.functional.normalize(mat, p=2, dim=1)

    @classmethod
    def _normalized_projections(cls, ps):
        n = ps.size(0) // 2
        h1, h2 = ps[:n], ps[n:]
        h2 = cls._norm(h2)
        h1 = cls._norm(h1)
        return n, h1, h2

    def _loss_func(self, ps):
        n, h1, h2 = self._normalized_projections(ps)

        labels = torch.arange(0, n, device=self.device, dtype=torch.long)
        masks = torch.eye(n, device=self.device)

        logits_aa = ((h1 @ h1.t()) / self.temp) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / self.temp) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / self.temp
        logits_ba = (h2 @ h1.t()) / self.temp

        loss_a = torch.nn.functional.cross_entropy(torch.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat((logits_ba, logits_bb), dim=1), labels)

        loss = (loss_a + loss_b)

        return loss

    def __call__(self, ps):
        return self._loss_func(ps)
    
