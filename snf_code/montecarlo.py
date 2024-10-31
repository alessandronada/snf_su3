import torch
import torch.utils.checkpoint as checkpoint
from snf_code.utils import *
from snf_code.distributions import *

def sum_of_staples(cfgs, mu, D, device):
    staple = torch.zeros(cfgs[:,0].shape, dtype=torch.cdouble).to(device)
    for nu in range(D):
        if nu != mu:
            pstaple = SUN_mul(torch.roll(cfgs,1,dims=(-D + mu - 2))[:,nu],
                              SUN_dagger(torch.roll(cfgs,1,dims=(-D + nu - 2)))[:,mu])
            pstaple = SUN_mul(pstaple, SUN_dagger(cfgs[:,nu]))
            staple += pstaple
            nstaple = SUN_mul(SUN_dagger(torch.roll(cfgs,(1,-1),dims=(-D + mu - 2,-D + nu - 2))[:,nu]),
                              SUN_dagger(torch.roll(cfgs,-1,dims=(-D + nu - 2)))[:,mu])
            nstaple = SUN_mul(nstaple, torch.roll(cfgs,-1,dims=(-D + nu - 2))[:,nu])
            staple += nstaple

    return staple

def SU2_element(mat, i, j):
    return torch.index_select(torch.index_select(mat,-2,i),-1,j).squeeze(-1).squeeze(-1)

def SU2toSU3(Ak, phb, device):
    A = torch.zeros(Ak.shape[:-2] + (2,1)).to(device)
    B = torch.cat((torch.index_select(Ak,-1,torch.arange(phb)),A,torch.index_select(Ak,-1,torch.arange(phb,2))),-1)
    A = torch.zeros(Ak.shape[:-2] + (1,2)).to(device)
    C = torch.ones(Ak.shape[:-2] + (1,1)).to(device)
    AA = torch.cat((torch.index_select(A,-1,torch.arange(phb)),C,torch.index_select(A,-1,torch.arange(phb,2))),-1)
    return torch.cat((torch.index_select(B,-2,torch.arange(phb)),AA,torch.index_select(B,-2,torch.arange(phb,2))),-2)


def heatbath(prefactor, mu, D, rand, device):
    #WARNING: if rand[:,0] and one of the two other rand are simultaneously 0 a NaN occurs in the gradient of mod. Probably very unlikely
    lam2 = (- 1.0 / (2.0 * prefactor) * (rand[:,0,mu,:] + rand[:,1,mu,:] * rand[:,2,mu,:]))
    acc = (1.0 - lam2 >= rand[:,3,mu,:]**2)

    x0 = (1.0 - 2.0 * lam2) * acc
    #x0=x0.detach() # This detach prevents (unlikely) NaN in the gradient of the square root mod
    stheta = torch.sqrt(1.0 - rand[:,4,mu,:]**2)
    mod = torch.sqrt(1.0 - x0**2)

    A = torch.complex(x0, mod*rand[:,4,mu,:]).unsqueeze(-1)
    B = (stheta*mod*torch.complex(torch.cos(rand[:,5,mu,:]), torch.sin(rand[:,5,mu,:]))).unsqueeze(-1)
    Xa = torch.ones(prefactor.shape + (1,)).to(device) * A
    Xb = torch.ones(prefactor.shape + (1,)).to(device) * B
    X = torch.cat((Xa,Xb),-1)

    v1 = torch.conj((X * torch.tensor([1,-1]).view([1]*(len(X.shape)-1)+[2])).roll(1,-1))
    Xmat = torch.cat((X.unsqueeze(-2),v1.unsqueeze(-2)),-2)
    acc = acc.unsqueeze(-1).unsqueeze(-1)

    return Xmat, acc

def SU2_link_update_hb(beta, cfgs, mu, D, rand, device):
    staples = sum_of_staples(cfgs, mu, D, device)
    k = torch.sqrt(SUN_determinant(staples).abs())
    staples = staples / k.unsqueeze(-1).unsqueeze(-1)

    Xmat, acc = heatbath(beta*k, mu, D, rand[:,0:6], device)
    new_cfgs = SUN_mul(Xmat, SUN_dagger(staples)) * acc + cfgs * ~acc
    return new_cfgs.unsqueeze(1)

def SU2_link_update_over(cfgs, mu, D, device):
    staples = sum_of_staples(cfgs, mu, D, device)
    k = torch.sqrt(SUN_determinant(staples).abs())
    staples = staples / k.unsqueeze(-1).unsqueeze(-1)

    staples = SUN_dagger(staples)
    new_cfgs = SUN_mul(SUN_mul(staples, SUN_dagger(cfgs)), staples)
    return new_cfgs.unsqueeze(1)

def SU3_compute_RK(new_cfgs, staples, sel, phb):
    sel2 = sel[sel != phb]
    Rk = SUN_mul(new_cfgs, staples)# SUN_dagger(staples))

    #generate SU(2) subgroup matrix from Rk
    a = 0.5*(SU2_element(Rk,sel2[0],sel2[0]) + torch.conj(SU2_element(Rk,sel2[1],sel2[1]))).unsqueeze(-1)
    b = 0.5*(SU2_element(Rk,sel2[0],sel2[1]) - torch.conj(SU2_element(Rk,sel2[1],sel2[0]))).unsqueeze(-1)
    v0 = torch.cat((a,b),-1)
    v1 = torch.conj((v0 * torch.tensor([1,-1]).view([1]*(len(v0.shape)-1)+[2])).roll(1,-1))
    Rk = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
    k = torch.sqrt(SUN_determinant(Rk).abs())
    Rk = Rk / k.unsqueeze(-1).unsqueeze(-1)
    return Rk, k

def SU3_link_update_hb(beta, cfgs, mu, D, rand, device):
    #staples = sum_of_staples(cfgs, mu, D, device)
    staples = checkpoint.checkpoint(sum_of_staples, cfgs, mu, D, device, use_reentrant=False)
    phb_steps = 3
    new_cfgs = cfgs[:,mu]
    sel = torch.tensor([0,1,2]).to(device)
    for phb in range(phb_steps):
        Rk, k = SU3_compute_RK(new_cfgs, staples, sel, phb)
        Xmat, acc = heatbath(2.0*beta*k/3.0, mu, D, rand[:,6*phb:6*(phb+1),:], device)
        Ak = SUN_mul(Xmat, SUN_dagger(Rk))
        Ak = SU2toSU3(Ak, phb, device)
        new_cfgs = SUN_mul(Ak, new_cfgs) * acc + new_cfgs * ~acc
    return new_cfgs.unsqueeze(1)

def SU3_link_update_over(cfgs, mu, D, device):
    staples = checkpoint.checkpoint(sum_of_staples, cfgs, mu, D, device, use_reentrant=False)
    phb_steps = 3
    new_cfgs = cfgs[:,mu]
    sel = torch.tensor([0, 1, 2])
    for phb in range(phb_steps):
        Rk, k = SU3_compute_RK(new_cfgs, staples, sel, phb)
        i0 = torch.tensor([0])
        i1 = torch.tensor([1])
        a = (SU2_element(Rk,i1,i1)*SU2_element(Rk,i1,i1)+SU2_element(Rk,i0,i1)*SU2_element(Rk,i1,i0)).unsqueeze(-1)
        b = (-SU2_element(Rk,i1,i1)*SU2_element(Rk,i0,i1)-SU2_element(Rk,i0,i1)*SU2_element(Rk,i0,i0)).unsqueeze(-1)
        v0 = torch.cat((a,b),-1)
        v1 = torch.conj((v0 * torch.tensor([1,-1]).view([1]*(len(v0.shape)-1)+[2])).roll(1,-1))
        Ak = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
        #bring Ak back to SU(3)
        Ak = SU2toSU3(Ak, phb, device)
        #new link is Ak*old_link
        new_cfgs = SUN_mul(Ak, new_cfgs)
    return new_cfgs.unsqueeze(1)

#haar measure sampling

def haar_heatbath(rand, cfgs_shape, device):
    x0 = rand[:,0,:]
    acc = (1.0 - x0**2 >= rand[:,3,:]**2)

    x0 = x0 * acc
    stheta = torch.sqrt(1.0 - rand[:,4,:]**2)
    mod = torch.sqrt(1.0 - x0**2)

    A = torch.complex(x0, mod*rand[:,4,:]).unsqueeze(-1)
    B = (stheta*mod*torch.complex(torch.cos(rand[:,5,:]), torch.sin(rand[:,5,:]))).unsqueeze(-1)
    Xa = torch.ones(cfgs_shape[:-2] + (1,)).to(device) * A
    Xb = torch.ones(cfgs_shape[:-2] + (1,)).to(device) * B
    v0 = torch.cat((Xa,Xb),-1)

    v1 = torch.conj((v0 * torch.tensor([1,-1]).view([1]*(len(v0.shape)-1)+[2])).roll(1,-1))
    Xmat = torch.cat((v0.unsqueeze(-2),v1.unsqueeze(-2)),-2)
    acc = acc.unsqueeze(-1).unsqueeze(-1)

    return Xmat, acc

def SU2_link_haar_hb(cfgs, rand, device):
    Xmat, acc = haar_heatbath(rand[:,0:6], cfgs.shape, device)
    new_cfgs = Xmat * acc + cfgs * ~acc
    return new_cfgs

def SU3_link_haar_hb(cfgs, rand, device):
    phb_steps = 3
    new_cfgs = cfgs
    for phb in range(phb_steps):
        Xmat, acc = haar_heatbath(rand[:,6*phb:6*(phb+1),:], cfgs.shape, device)
        Ak = SU2toSU3(Xmat, phb, device)
        new_cfgs = SUN_mul(Ak, new_cfgs) * acc + new_cfgs * ~acc

    return new_cfgs

def init_hb(N, mask, D, orsteps, rn_shape, init_shape):
    if N==2:
        hb_update=SU2_link_update_hb
        haar_hb_update=SU2_link_haar_hb
        over_update=SU2_link_update_over
    elif N==3:
        hb_update=SU3_link_update_hb
        haar_hb_update=SU3_link_haar_hb
        over_update=SU3_link_update_over

    def update(cfgs, beta, device):
        rand = torch.rand(rn_shape).to(device)
        if beta == 0.0:
            for rr in range(rn_shape[1]//6):
                rand[:,rr*6] = 1.0 - 2.0 * rand[:,rr*6]
                rand[:,rr*6+4] = 1.0 - 2.0 * rand[:,rr*6+4]
                rand[:,rr*6+5] = 2.0 * torch.pi * rand[:,rr*6+5]

            cfgs = haar_hb_update(cfgs, rand, device)
        else:
            for rr in range(rn_shape[1]//6):
                rand[:,rr*6] = torch.log(1.0 - rand[:,rr*6])
                rand[:,rr*6+1] = torch.cos(2.0 * torch.pi *(1.0 - rand[:,rr*6+1]))**2
                rand[:,rr*6+2] = torch.log(1.0 - rand[:,rr*6+2])
                rand[:,rr*6+4] = 1.0 - 2.0 * rand[:,rr*6+4]
                rand[:,rr*6+5] = 2.0 * torch.pi * rand[:,rr*6+5]

            for mu in range(D):
                for eo in range(2):
                    current_mask = mask[eo+mu*2,:]
                    cfgs_new = hb_update(beta, cfgs, mu, D, rand, device)
                    cfgs = current_mask * cfgs_new + (1-current_mask) * cfgs

            for o in range(orsteps):
                for mu in range(D):
                    for eo in range(2):
                        current_mask = mask[eo+mu*2,:]
                        cfgs_new = over_update(cfgs, mu, D, device)
                        cfgs = current_mask * cfgs_new + (1-current_mask) * cfgs
        return cfgs

    return update


class NED_update(torch.nn.Module):
    def __init__(self, action, beta, update, device, nupdates=1):
        super().__init__()
        self.action = action
        self.update = update
        self.beta = beta
        self.nupdates = nupdates
        self.device = device

    def forward(self, cfgs):
        s_old = self.action(cfgs, self.beta)
        for u in range(self.nupdates):
            cfgs = self.update(cfgs, self.beta, self.device)
        return cfgs, self.action(cfgs, self.beta) - s_old, 0
