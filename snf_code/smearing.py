import torch
import torch.utils.checkpoint as checkpoint
from snf_code.utils import *
from snf_code.distributions import *

#functions for smearing
def xi0(w, w2):
    return torch.where(torch.abs(w) > 0.005, torch.sin(w)/w, 1. - 1./6.*w2*(1. - 1./20.*w2*(1. - 1./42.*w2)))

def xi1(w, w2):
    return torch.where(torch.abs(w) > 0.005, torch.cos(w)/w2 - torch.sin(w)/w**3,
                       -1./3. + w2*(1./30. + w2*(-1./840 + 1./45360.*w2)))

def xi2(w, w2, xizero, xione):
    return torch.where(torch.abs(w) > 0.005, 1./w2*(xizero + 3. * xione), -1./15. + w2*(1./210. - w2/7560.))

#stuff for Jacobian
def otimes(A,B):
    return torch.einsum('...ij,...kl->...ijkl',A,B)

def oplus(A,B):
    return torch.einsum('...kj,...il->...ijkl',A,B)

def starprod(A,B):
    return torch.einsum('...inml,...njkm->...ijkl',A,B)

def starprodmat(A,B):
    return torch.einsum('...ijkn,...nl->...ijkl',A,B)

def matstarprod(A,B):
    return torch.einsum('...in,...njkl->...ijkl',A,B)

def generate_coefficients(Q, Q2, id, oidid, oidQ, device):
    c0 = SUN_determinant(Q)
    c1 = .5 * SUN_trace(Q2)
    c0max = 2.0 * (c1 / 3.0)**1.5

    sgnc0 = torch.real(torch.sgn(c0))
    c0 = torch.abs(c0)

    theta = torch.arccos(c0/c0max)
    u = torch.sqrt(c1/3.0) * torch.cos(theta/3.0)
    w = torch.sqrt(c1) * torch.sin(theta/3.0)
    u2 = u**2
    w2 = w**2
    cw = torch.cos(w)
    eu = torch.cos(u) + 1.j*torch.sin(u)
    eu2 = eu**2
    eum = eu**(-1)

    xizero = xi0(w, w2)
    xione = xi1(w, w2)

    h0 = (u2 - w2)*eu2 + eum*(8.*u2*cw + 2.j*u*(3.*u2 + w2)*xizero)
    h1 = 2.*u*eu2 - eum*(2.*u*cw - 1.j*(3.*u2 - w2)*xizero)
    h2 = eu2 - eum*(cw + 3.j*u*xizero)

    r10 = 2.*(u + 1.j*(u2 - w2))*eu2 + 2.*eum*(4.*u*(2.-1.j*u)*cw + 1.j*(9.*u2 + w2 - 1.j*u*(3.*u2 + w2)) * xizero)
    r11 = 2.*(1. + 2.j*u)*eu2 + eum*(-2.*(1.-1.j*u)*cw + 1.j*(6.*u + 1.j*(w2 - 3.*u2))*xizero)
    r12 = 2.j*eu2 + 1.j*eum*(cw - 3.*(1.-1.j*u)*xizero)
    r20 = -2.*eu2 + 2.j*u*eum*(cw + (1.+4.j*u)*xizero + 3.*u2*xione)
    r21 = -1.j*eum*(cw + (1.+2.j*u)*xizero - 3.*u2*xione)
    r22 = eum * (xizero - 3.j*u*xione)

    den = 9.*u2 - w2
    den2 = 2. * den**2
    v3u2mw2 = 3.*u2 - w2
    v15u2pw2 = 15.*u2 + w2

    f0 = h0 / den
    f1 = h1 / den
    f2 = h2 / den

    b10 = (2.*u*r10 + v3u2mw2 * r20 - 2.*v15u2pw2*f0) / den2
    b11 = (2.*u*r11 + v3u2mw2 * r21 - 2.*v15u2pw2*f1) / den2
    b12 = (2.*u*r12 + v3u2mw2 * r22 - 2.*v15u2pw2*f2) / den2
    b20 = (r10 - 3.*u*r20 - 24.*u*f0) / den2
    b21 = (r11 - 3.*u*r21 - 24.*u*f1) / den2
    b22 = (r12 - 3.*u*r22 - 24.*u*f2) / den2

    f0 = torch.where(sgnc0 > 0, f0, torch.conj(f0))
    f1 = torch.where(sgnc0 > 0, f1, -torch.conj(f1))
    f2 = torch.where(sgnc0 > 0, f2, torch.conj(f2))

    b10 = torch.where(sgnc0 > 0, b10, torch.conj(b10))
    b11 = torch.where(sgnc0 > 0, b11, -torch.conj(b11))
    b12 = torch.where(sgnc0 > 0, b12, torch.conj(b12))
    b20 = torch.where(sgnc0 > 0, b20, -torch.conj(b20))
    b21 = torch.where(sgnc0 > 0, b21, torch.conj(b21))
    b22 = torch.where(sgnc0 > 0, b22, -torch.conj(b22))

    B1 = b10.unsqueeze(-1).unsqueeze(-1) * id + b11.unsqueeze(-1).unsqueeze(-1) * Q +\
         + b12.unsqueeze(-1).unsqueeze(-1) * Q2
    B2 = b20.unsqueeze(-1).unsqueeze(-1) * id + b21.unsqueeze(-1).unsqueeze(-1) * Q +\
         + b22.unsqueeze(-1).unsqueeze(-1) * Q2

    return f0.unsqueeze(-1).unsqueeze(-1), f1.unsqueeze(-1).unsqueeze(-1), f2.unsqueeze(-1).unsqueeze(-1), B1, B2

def generate_omega(C, U):
    return SUN_mul(C, SUN_dagger(U))

def generate_Q(omega, N):   #check i in definition
    return 0.5 * 1.j * (SUN_dagger(omega) - omega) - 0.5 * 1.j/N * SUN_trace(SUN_dagger(omega) - omega).unsqueeze(-1).unsqueeze(-1)

def generate_expQ(Q, Q2, id, f0, f1, f2):
    return f0 * id + f1 * Q + f2 * Q2

def generate_dQ_domega(N, oidid, id):
    A = oidid
    B = oplus(id, id)
    return -1.j*(.5 * A - .5/N * B)

def generate_dexpQ_dQ(Q, Q2, B1, B2, f1, f2, id, oidid, oidQ):
    M = oplus(Q, B1)
    M += oplus(Q2, B2)
    M += f1.unsqueeze(-1).unsqueeze(-1)*oidid
    M += f2.unsqueeze(-1).unsqueeze(-1)*oidQ
    return M

def generate_domega_dU(C, id):
    return -otimes(id, SUN_dagger(C))

def generate_domega_dC(U, oidid):
    return starprodmat(oidid, SUN_dagger(U))

def generate_dQ_dU(dQdomega, id, C):
    B = generate_domega_dU(C, id)
    return starprod(dQdomega, B)

def generate_dQ_dC(dQdomega, oidid, U):
    B = generate_domega_dC(U, oidid)
    return starprod(dQdomega, B)

def generate_dexpQ_dU(dexpQdQ, dQ_dU):
    return starprod(dexpQdQ, dQ_dU)

def generate_Jacobian_U(dexpQdU, U, expQ, oidid):
    return starprodmat(dexpQdU, U) + matstarprod(expQ, oidid)

def generate_Jacobian_C(dexpQdQ, dQdC, U):
    A = starprod(dexpQdQ, dQdC)
    return starprodmat(A, U)

def Jacobian_reshape(jac, jac_shape):
    return torch.einsum('...ijkl->...iljk',jac).reshape(jac_shape)

def det_Jac(J):
    return torch.linalg.det(J)

def stout_staples(cfgs, mu, D, rho, device):
    staple = torch.zeros(cfgs[:,0].shape, dtype=torch.cdouble).to(device)
    if len(rho) == 1:
        for nu in range(D):
            if nu != mu:
                staple += rho * (SUN_dagger(pstaple(cfgs, mu, nu, D)) + SUN_dagger(nstaple(cfgs, mu, nu, D)))
    else:
        for nu in range(D):
            if nu != mu:
                staple += rho[nu] * (SUN_dagger(pstaple(cfgs, mu, nu, D)) + SUN_dagger(nstaple(cfgs, mu, nu, D)))
    return staple   

def stout_smearing(cfgs, mu, D, rho, jac_shape, device):
    C = checkpoint.checkpoint(stout_staples, cfgs, mu, D, rho, device, use_reentrant=False, preserve_rng_state=False)
    
    U = cfgs[:,mu]
    id = SUN_identity(U.shape[:-1])
    N = U.shape[-1]
    omega = generate_omega(C, U)
    Q = generate_Q(omega, N)
    Q2 = SUN_mul(Q, Q)

    oidid = otimes(id, id)
    oidQ = otimes(id, Q) + otimes(Q, id)

    f0, f1, f2, B1, B2 = generate_coefficients(Q, Q2, id, oidid, oidQ, device)
    expQ = generate_expQ(Q, Q2, id, f0, f1, f2)
    dexpQdQ = generate_dexpQ_dQ(Q, Q2, B1, B2, f1, f2, id, oidid, oidQ)

    dQdomega = generate_dQ_domega(N, oidid, id)
    dQdU = generate_dQ_dU(dQdomega, id, C)
    dexpQdU = generate_dexpQ_dU(dexpQdQ, dQdU)
    Jacobian_U = generate_Jacobian_U(dexpQdU, U, expQ, oidid)
    detjac = det_Jac(Jacobian_reshape(Jacobian_U, jac_shape)).abs()
    
    return SUN_mul(expQ, U).unsqueeze(1), detjac.unsqueeze(1)

class Smearing(torch.nn.Module):
    def __init__(self, N, mask, D,  jac_shape, scale_fac, rho_shape, device, rho_root=None, smearing_steps=1):
        super().__init__()
        self.N = N
        self.mask = mask
        self.D = D
        self.smearing_steps = smearing_steps
        
        if rho_root is None:
            self.rho_root = torch.nn.Parameter(torch.rand(rho_shape).to(device) * scale_fac)
        else:
            self.rho_root = torch.nn.Parameter(rho_root)

        self.jac_shape = jac_shape
        self.device = device

    def forward(self, cfgs):
        ones = (torch.ones(self.mask[0].shape, dtype=torch.double).squeeze(-1).squeeze(-1)).to(self.device)
        bs = cfgs.shape[0]
        dlogJ = torch.zeros(bs, dtype=torch.double).to(self.device)
        dims = tuple(np.arange(1,self.D+2))
        for sm in range(self.smearing_steps):
            for mu in range(self.D):
                for eo in range(2):
                    current_mask = self.mask[eo+mu*2,:]
                    rho = self.rho_root[sm,eo+mu*2,:]**2
                    cfgs_new, jac = stout_smearing(cfgs, mu, self.D, rho, self.jac_shape, self.device)
                    cfgs = current_mask * cfgs_new + (1-current_mask) * cfgs
                    jac = torch.log(current_mask.squeeze(-1).squeeze(-1) * jac + (1-current_mask.squeeze(-1).squeeze(-1)) * ones)
                    dlogJ += torch.sum(jac, dims)
        return cfgs, 0, dlogJ
