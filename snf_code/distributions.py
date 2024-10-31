import torch
from snf_code.utils import *


class PriorSUN:
    def __init__(self, init_shape, action, beta, update, therm_steps, mcmc_steps, device):
        self.init_shape=init_shape
        self.update=update
        self.beta=beta
        self.therm_steps=therm_steps
        self.mcmc_steps=mcmc_steps
        self.action=action
        self.device=device

    def __call__(self, cfgs=None):
        if cfgs != None:
            for i in range(self.mcmc_steps):
                cfgs = self.update(cfgs, self.beta, self.device)
        else:
            cfgs = self.therm()
        return cfgs, self.action(cfgs, self.beta)

    def therm(self):
        cfgs = SUN_identity(self.init_shape).to(self.device)
        for i in range(self.therm_steps):
            cfgs = self.update(cfgs, self.beta, self.device)
        return cfgs


class S_SUN:
    def __init__(self, D,N):
      self.D=D
      self.N=N
    def __call__(self,cfgs, beta):
      retr = torch.zeros(cfgs[:,0].shape[:-2], dtype = torch.double)
      dims = range(1,self.D+1)
      for nu in range(1,self.D):
          for mu in range(0,nu):
              plaq = SUN_mul(SUN_dagger(cfgs[:,nu]),cfgs[:,mu])
              plaq = SUN_mul(plaq,torch.roll(cfgs,1,dims=(-self.D + mu - 2))[:,nu])
              plaq = SUN_mul(plaq,SUN_dagger(torch.roll(cfgs,1,dims=(-self.D + nu - 2)))[:,mu])
              retr += SUN_trace(plaq).real
      return beta*torch.sum(6.0 - retr/self.N, tuple(dims))

