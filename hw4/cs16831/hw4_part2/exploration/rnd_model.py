from cs16831.hw4_part2.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import ipdb;

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec
        
        # <TODO>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        self.f = ptu.build_mlp(input_size=self.ob_dim,
                               output_size=self.output_size,
                               n_layers=self.n_layers,
                               size=self.size,
                               init_method=init_method_1)
        self.f.to(ptu.device)
        # 2) f_hat, the function we are using to learn f
        self.f_hat = ptu.build_mlp(input_size=self.ob_dim,
                                   output_size=self.output_size,
                                   n_layers=self.n_layers,
                                   size=self.size,
                                   init_method=init_method_2)
        self.f_hat.to(ptu.device)
        self.f_hat_optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )

    def forward(self, ob_no):
        # <TODO>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        if type(ob_no) != torch.Tensor:
            ob_no = ptu.from_numpy(ob_no)
        f_encoding = self.f(ob_no).detach()
        f_hat_encoding = self.f_hat(ob_no)
        
        error = ((f_encoding - f_hat_encoding) ** 2).sum(dim=1)
        return error # shape (batch_size,)

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <TODO>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        if type(ob_no) != torch.Tensor:
            ob_no = ptu.from_numpy(ob_no)

        error = self(ob_no).mean()  # mean error across the batch shape (1,)
        self.f_hat_optimizer.zero_grad()
        error.backward()
        self.f_hat_optimizer.step()
        return error.item()
