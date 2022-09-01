"""
Model trainer.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from apex import amp

from utils import clustering_by_representation, seed_everything, print_network, classification_metric
from optimizer import get_optimizer


class Trainer:
    
    def __init__(self, 
                 model_cls, 
                 args, 
                 train_dataset, 
                 valid_dataset=None, 
                 test_dataset=None,
                 device='cpu') -> None:
        self.num_workers = args.train.num_workers
        self.device = device
        # unsupervised or classification.
        self.train_type = args.task
        assert self.train_type in ['clustering', 'classification']
        self.model_cls = model_cls
        self.args = args
        self.save_embeddings = self.args.fusion.save_embeddings
        self.train_dataset = train_dataset
        self.history = {
            'seeds': []
        }
        if self.save_embeddings != -1:
            self.test_batch = torch.load('./experiments/results/test_batch')
            self.history['test_labels'] = self.test_batch['y']
            
        self.valid_dataset = valid_dataset
        if self.args.task == 'classification':
            self.test_dataset = test_dataset
        self.writer = SummaryWriter(log_dir=self.args.train.log_dir)
        with open(f"{self.args.train.log_dir}/hparams.yaml", 'w') as f:
            f.write(args.dump())
        seed_everything(args.seed)
        
        
    def train(self):
        
        if self.args.task == 'clustering':
            self._train_clustering()
        else:
            self._train_classification()
        
    def _train_classification(self):
        for t in range(self.args.test_time):
            seed = torch.randint(10000+t+1, (1, )).item()
            seed_everything(seed)
            self.history['seeds'].append(seed)
            self.history[f"seed:{seed}"] = {"acc": [],"P": [], "f_score": [], "loss": []}

            model = self.model_cls(self.args, 
                                   device=self.device)
            model = model.to(self.device, non_blocking=True)
            optimizer = get_optimizer(model.parameters(), lr=self.args.train.lr, type=self.args.train.optim)
            if self.args.train.fp16:
                model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.train.opt_level)
                print(f'Enable mixed precision.')
            
            if self.args.verbose:
                print_network(model)
            train_loader = DataLoader(self.train_dataset, 
                                      self.args.train.batch_size, 
                                      num_workers=self.num_workers,
                                      shuffle=True, 
                                      pin_memory=True,
                                      drop_last=True)
            valid_loader = DataLoader(self.valid_dataset, 
                                      self.args.train.batch_size * 2, 
                                      num_workers=self.num_workers)
            
            test_loader = DataLoader(self.test_dataset, 
                                      self.args.train.batch_size * 2, 
                                      num_workers=self.num_workers)
            
            pretrain_epochs = self.args.train.epochs - 10
            train_epochs = 10
            
            for epoch in range(pretrain_epochs):
                model.train()
                losses = []
                if self.args.verbose:
                    pbar = tqdm(total=len(train_loader), ncols=0, unit=" batch")
                for data in train_loader:
                    [x1, x2], y = data
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    y = y.to(self.device)
                    loss, _ = model.get_loss([x1, x2])
                    
                    optimizer.zero_grad()
                    if self.args.train.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if self.args.verbose:
                        pbar.update()
                        pbar.set_postfix(
                            tag='UNSUP',
                            epoch=epoch,
                            loss=f"{np.mean(losses):.4f}",
                        )
                self.writer.add_scalars('unsup-train-loss', {f'seed:{seed}': np.mean(losses)}, global_step=epoch)
                self.history[f'seed:{seed}']['loss'].append(np.mean(losses))
                if self.args.verbose:
                    pbar.close()
                    
                # can use valid dataloader to save the base model.
            for epoch in range(train_epochs):
                model.train()
                losses = []
                if self.args.verbose:
                    pbar = tqdm(total=len(train_loader), ncols=0, unit=" batch")
                for data in train_loader:
                    [x1, x2], y = data
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    y = y.to(self.device)
                    loss = model.get_loss([x1, x2], y)
                    
                    optimizer.zero_grad()
                    if self.args.train.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if self.args.verbose:
                        pbar.update()
                        pbar.set_postfix(
                            tag='SUP',
                            epoch=epoch,
                            loss=f"{np.mean(losses):.4f}",
                        )
                self.writer.add_scalars('sup-train-loss', {f'seed:{seed}': np.mean(losses)}, global_step=epoch)
                self.history[f'seed:{seed}']['loss'].append(np.mean(losses))
                if self.args.verbose:
                    pbar.close()   
                    
                if test_loader:
                    model.eval()
                    predicts = []
                    ground_truth = []
                    with torch.no_grad():
                        for data in test_loader:
                            [x1, x2], y = data
                            x1, x2 = x1.to(self.device), x2.to(self.device)
                            y = y.to(self.device)
                            ground_truth.append(y.detach().cpu())
                            pred = model.predict([x1, x2])
                            predicts.append(pred)
                        ground_truth = torch.concat(ground_truth, dim=-1).numpy()
                        predicts = torch.vstack(predicts).squeeze().detach().cpu().numpy()
                        acc, f_score, precision = classification_metric(ground_truth, predicts)
                        if self.args.verbose:
                            print(f"[Valid] ACC: {acc}, P: {precision}, f_score: {f_score}")
                        # Record.
                        self.writer.add_scalars('test-acc', {f'seed:{seed}': acc}, global_step=epoch)
                        self.writer.add_scalars('test-p', {f'seed:{seed}': precision}, global_step=epoch)
                        self.writer.add_scalars('test-fscore', {f'seed:{seed}': f_score}, global_step=epoch)
                        self.history[f'seed:{seed}']['acc'].append(acc)
                        self.history[f'seed:{seed}']['P'].append(precision)
                        self.history[f'seed:{seed}']['f_score'].append(f_score)
                
        if self.args.train.save_log:
            torch.save(self.history, f"{self.args.train.log_dir}/history.log")
        self.writer.close()
         
                    
            
    
    def _train_clustering(self):
        for t in range(self.args.test_time):
            seed = torch.randint(10000+t+1, (1, )).item()
            seed_everything(seed)
            self.history['seeds'].append(seed)
            self.history[f"seed:{seed}"] = {"acc": [],"nmi": [], "ari": [], "loss": [], "class_acc": [], "p": [], 'fscore': []}
            
            model = self.model_cls(self.args, 
                                   device=self.device)
            model = model.to(self.device, non_blocking=True)
            optimizer = get_optimizer(model.parameters(), lr=self.args.train.lr, type=self.args.train.optim)
            
            if self.args.train.fp16:
                model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.train.opt_level)
                print(f'Enable mixed precision.')
            
            if self.args.verbose:
                print_network(model)
            train_loader = DataLoader(self.train_dataset, 
                                      self.args.train.batch_size, 
                                      num_workers=self.num_workers,
                                      shuffle=True, 
                                      pin_memory=True,
                                      drop_last=True)
            valid_loader = DataLoader(self.valid_dataset, 
                                      self.args.train.batch_size * 2, 
                                      num_workers=self.num_workers) if self.valid_dataset else None
            for epoch in range(self.args.train.epochs):
                model.train()
                losses = []
                if self.args.verbose:
                    pbar = tqdm(total=len(train_loader), ncols=0, unit=" batch")
                for data in train_loader:
                    [x1, x2], y = data
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                    
                    loss, _ = model.get_loss([x1, x2])
                    
                    optimizer.zero_grad()
                    if self.args.train.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    
                    losses.append(loss.item())
                    
                    if self.args.verbose:
                        pbar.update()
                        pbar.set_postfix(
                            epoch=epoch,
                            loss=f"{np.mean(losses):.4f}",
                        )
                self.writer.add_scalars('train-loss', {f'seed:{seed}': np.mean(losses)}, global_step=epoch)
                self.history[f'seed:{seed}']['loss'].append(np.mean(losses))
                if self.args.verbose:
                    pbar.close()
                
                if valid_loader:
                    model.eval()
                    repr = []
                    ground_truth = []
                    with torch.no_grad():
                        for data in valid_loader:
                            [x1, x2], y = data
                            x1, x2 = x1.to(self.device), x2.to(self.device)
                            ground_truth.append(y)
                            Z = model.commonZ([x1, x2])
                            repr.append(Z)
                        ground_truth = torch.concat(ground_truth, dim=-1).numpy()
                        repr = torch.vstack(repr).detach().cpu().numpy()
                        
                        acc, nmi, ari, class_acc, p, fscore = clustering_by_representation(repr, ground_truth)
                        if self.args.verbose:
                            print(f"[Valid] ACC: {acc}, NMI: {nmi}, ARI: {ari}, class_ACC: {class_acc}, p: {p}, fscore: {fscore}")
                        # Record.
                        self.writer.add_scalars('valid-acc', {f'seed:{seed}': acc}, global_step=epoch)
                        self.writer.add_scalars('valid-nmi', {f'seed:{seed}': nmi}, global_step=epoch)
                        self.writer.add_scalars('valid-ari', {f'seed:{seed}': ari}, global_step=epoch)
                        self.history[f'seed:{seed}']['acc'].append(acc)
                        self.history[f'seed:{seed}']['nmi'].append(nmi)
                        self.history[f'seed:{seed}']['ari'].append(ari)
                        
                        self.writer.add_scalars('valid-class-acc', {f'seed:{seed}': acc}, global_step=epoch)
                        self.writer.add_scalars('valid-p', {f'seed:{seed}': p}, global_step=epoch)
                        self.writer.add_scalars('valid-fscore', {f'seed:{seed}': fscore}, global_step=epoch)
                        self.history[f'seed:{seed}']['class_acc'].append(class_acc)
                        self.history[f'seed:{seed}']['p'].append(p)
                        self.history[f'seed:{seed}']['fscore'].append(fscore)
                        
                    if self.save_embeddings != -1 and epoch % self.save_embeddings == 0:
                        [x1, x2], y = self.test_batch
                        x1, x2 = x1.to(self.device), x2.to(self.device)
                        hs, z = model.extract_all_hidden([x1, x2])
                        hs = [h.detach().cpu() for h in hs]
                        z = z.detach().cpu()
                        self.history[f'seed:{seed}'][f'hidden_hs_{epoch}'] = hs
                        self.history[f'seed:{seed}'][f'hidden_z_{epoch}'] = z
                
        if self.args.train.save_log:
            torch.save(self.history, f"{self.args.train.log_dir}/history.log")
        self.writer.close()
        
        
        
        
    
    