import torch
import numpy as np
import time
from scipy.optimize import fsolve

### flow utils

def get_lr(optimizer):
    for p in optimizer.param_groups:
        return p["lr"]

def grab(var):
    if torch.is_tensor(var):
        return var.detach().cpu().numpy()
    else:
        return var

def save(model, optimizers, path):
    opts_sd = []
    for opt in optimizers:
        opts_sd.append(opt.state_dict())

    torch.save({'model_state_dict': model.state_dict(), 'optimizers_state_dict': opts_sd}, path)

def load(model, optimizers, path, device='cuda'):
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    opts_sd = checkpoint['optimizers_state_dict']
    i = 0
    for opt in optimizers:
        opt.load_state_dict(opts_sd[i])
        i += 1

def write(history,root):
    ess_file = root + '_ESS.dat'
    loss_var_file = root + '_lossvar.dat'
    loss_file = root + '_loss.dat'

    with open(ess_file, 'w') as f:
        for item in history['ESS']:
            f.write("%f\n" % item)

    with open(loss_var_file, 'w') as f:
        for item in history['var_loss']:
            f.write("%f\n" % item)

    with open(loss_file, 'w') as f:
        for item in history['loss']:
            f.write("%f\n" % item)

def print_metrics(history_file,history, era, epoch, avg_last_N_epochs, t0):
    with open(history_file, 'a') as f:
        f.write(f'\n == Era {era} | Epoch {epoch} metrics ==\n')
        for key, val in history.items():
            avgd = np.mean(val[-avg_last_N_epochs:])
            f.write(f'\t{key} {avgd:g}\n')
        f.write(str(time.time()-t0))

### Scale setting Necco-Sommer 5.7<beta<6.92

def ar0(beta):
    return np.exp( - 1.6804 - 1.7331 * (beta - 6.0) + 0.7849 * (beta - 6.0)**2 - 0.4428 * (beta - 6.0)**3)

def betafromar0(ar0_input):
    return fsolve(lambda beta : ar0_input - ar0(beta) , 6.0*np.ones(len(ar0_input)))

### Initialize rho from fit results

def rho_fit(x, a, b, d):
    #return a*np.tanh(b*x) + c - d*x
    return a*np.tanh(b*x) - d*x

def set_rho(beta_steps, beta0, betat, rho_shape):
    rho = []
       
    if beta0==5.587 and betat==5.688:
        a=0.00666252
        b=0.21683116 
        d=0.00115253
    elif beta0==5.756 and betat==5.877:
        a=0.00461724 
        b=0.29896732 
        d=0.00040807
    elif beta0==5.896 and betat==6.037:
        a=4.13554386e-03 
        b=7.08816138e-01 
        d=3.98324071e-04
    elif beta0==6.02 and betat==6.178:
        a=0.00417921
        b=0.70224363
        d=0.00074882
    else:
        return 0

    for bl in range(beta_steps):
        x = (bl+1)/beta_steps
        rho.append(torch.sqrt(torch.tensor(rho_fit(x, a, b*beta_steps, d)/beta_steps) * torch.ones(rho_shape)))
        #rho.append(torch.sqrt(torch.tensor(rho_fit(x, a, b*beta_steps, c0 + c1*beta_steps, d)/beta_steps) * torch.ones(rho_shape)))

    return rho

### SU(N) utils

def SUN_identity(init_shape):
    id = torch.ones(init_shape, dtype=torch.cdouble)
    return torch.diag_embed(id)

def hot_SU2_start_nonuniform(init_shape_SU2):
    x0 = torch.rand(init_shape_SU2, dtype=torch.cdouble)
    nrm = torch.norm(x0, dim = -1)
    x0 = x0 / nrm.unsqueeze(-1)

    x1 = torch.conj((x0 * torch.tensor([1,-1]).view([1]*(len(init_shape_SU2)-1)+[2])).roll(1,-1))

    return torch.cat((x0.unsqueeze(-2),x1.unsqueeze(-2)),-2)

def SUN_determinant(cfgs):
    return torch.linalg.det(cfgs)

def SUN_trace(cfgs):
    return torch.einsum('...ii', cfgs)

def check_determinant_SUN(cfgs, tol):
    dtm = SUN_determinant(cfgs)
    return (dtm - 1.0).abs() > tol

def SUN_dagger(cfgs):
    return torch.conj(torch.transpose(cfgs,-2,-1))

def SUN_mul(U,V):
    return torch.einsum('...ij,...jk->...ik',U,V)

def plaquette_SUN(cfgs, D, N):
    retr = torch.zeros(cfgs[:,0].shape[:-2], dtype = torch.double)
    dims = range(1,D+1)
    for nu in range(1,D):
        for mu in range(0,nu):
            plaq = SUN_mul(SUN_dagger(cfgs[:,nu]),cfgs[:,mu])
            plaq = SUN_mul(plaq,torch.roll(cfgs,1,dims=(-D + mu - 2))[:,nu])
            plaq = SUN_mul(plaq,SUN_dagger(torch.roll(cfgs,1,dims=(-D + nu - 2)))[:,mu])
            retr += SUN_trace(plaq).real
    return torch.mean(retr, tuple(dims))/(D*(D-1)/2)/N


def create_mask(D, T, L):
    mask_shape = (2*D,D,T)
    for i in range(1,D):
        mask_shape += (L,)

    mask = torch.zeros(mask_shape, dtype = int)
    for mu in range(D):
        for t in range(T):
            for x in range(L):
                if(D>2):
                    for y in range(L):
                        if(D>3):
                            for z in range(L):
                                parity = t + x + y + z
                                mask[parity%2+mu*2,mu,t,x,y,z] = 1
                        else:
                            parity = t + x + y
                            mask[parity%2+mu*2,mu,t,x,y] = 1
                else:
                    parity = t + x
                    mask[parity%2+mu*2,mu,t,x] = 1

    return mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

def pstaple(cfgs, mu, nu, D):
    pstaple = SUN_mul(torch.roll(cfgs,1,dims=(-D + mu - 2))[:,nu], SUN_dagger(torch.roll(cfgs,1,dims=(-D + nu - 2)))[:,mu])
    return SUN_mul(pstaple, SUN_dagger(cfgs[:,nu]))

def nstaple(cfgs, mu, nu, D):
    nstaple = SUN_mul(SUN_dagger(torch.roll(cfgs,(1,-1),dims=(-D + mu - 2,-D + nu - 2))[:,nu]), SUN_dagger(torch.roll(cfgs,-1,dims=(-D + nu - 2)))[:,mu])
    return SUN_mul(nstaple, torch.roll(cfgs,-1,dims=(-D + nu - 2))[:,nu])
