from typing import Optional, Tuple

import torch
from torch import Tensor

from torch.optim import Optimizer
from torch.optim import lr_scheduler
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from tqdm import trange #weird flex but still feel okay with that

from utils import permute_data
from model import Model

class Trainer(object): 
    def __init__(self,
                 model: Model,
                 optim: Optimizer,
                 criterion: _Loss):
        self.model = model
        self.optim = optim
        self.loss = criterion 
        self._check_optim_net_aligned()
    
    def _check_optim_net_aligned(self):
        assert self.optim.param_groups[0]['params'] == list(self.model.parameters()), \
            '''
            Ensure the optimization got the same input with model
            '''

    def _generate_batches(self, 
                          X: Tensor,
                          y: Tensor,
                          size: int = 32) -> Tuple[Tensor]:
        N = X.shape[0]

        for ii in range(0, N, size): 
            X_batch, y_batch = X[ii:ii+size], y[ii:ii+size]
            yield X_batch, y_batch

    def fit(self,
            X_train: Optional[Tensor],
            y_train: Optional[Tensor],
            X_test: Optional[Tensor],
            y_test: Optional[Tensor],
            train_loader: DataLoader = None,
            test_loader: DataLoader = None, 
            epochs: int = 1000,
            eval_every: int=10,
            batch_size: int=32,
            final_lr_exp: int = None):
        init_lr = self.optim.param_groups[0]['lr']

        if final_lr_exp:
            decay = (final_lr_exp / init_lr) ** (1 / (epochs + 1))
            scheduler = lr_scheduler.ExponentialLR(self.optim, gamma=decay)

            for e in (e := trange(epochs)): 
                if final_lr_exp:
                    scheduler.step()
                
                if not train_loader:
                    X_train, y_train = permute_data(X_train, y_train)

                    batch_generator = self._generate_batches(X_train, y_train, batch_size)

                    self.model.train()

                for ii, (X_batch, y_batch) in enumerate(batch_generator):
                    self.optim.zero_grad()

                    output = self.model(X_batch)[0]

                    loss = self.loss(output, y_batch)

                    loss.backward()
                    self.optim.step()

                if e % eval_every == 0: 
                    with torch.no_grad():
                        self.model.eval()
                        losses = []
                        for X_batch, y_batch in enumerate(test_loader):
                            output = self.model(X_batch)[0]
                            loss = self.loss(output, y_batch)
                            losses.append(loss.item())
                            v = round(torch.Tensor(losses).mean().item(), 5)
                        print("the loss after {0} epochs is was {1}".format(e, v))