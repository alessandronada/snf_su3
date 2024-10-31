import torch
from snf_code.montecarlo import *
from snf_code.smearing import *

class Flow(torch.nn.Module):
    def __init__(self, action, couplings, layers):
        super().__init__()
        self.action = action
        self.layers = layers
        self.couplings = couplings

    def __call__(self, x, s0, int_meas=1):
        x, Q, logJ, int_obs = self.forward(x, s0, int_meas=int_meas)
        st = self.action(x, self.couplings[-1])
        w = st - s0 - Q - 2.0*logJ
        return x, w, st, Q, 2.0*logJ, int_obs

    def forward(self, x, s0, int_meas=1):
        batch_size = x.shape[0]
        Q = torch.zeros(batch_size)
        logJ = torch.zeros(batch_size)
        int_obs = torch.zeros((batch_size, int_meas, 5))
        
        l = 0
        m = 0
        for layer in self.layers:
            x, dQ, dlogJ = layer.forward(x)
            Q += dQ
            logJ += dlogJ
            # index l runs on stochastic updates
            if isinstance(layer, NED_update):
                if (l+1) % (len(self.couplings) // int_meas) == 0:
                    st = self.action(x, self.couplings[l])
                    int_obs[:,m,0] = st - s0 - Q - 2.0*logJ #work
                    int_obs[:,m,1] = Q #heat
                    int_obs[:,m,2] = 2.0*logJ #jacobian
                    int_obs[:,m,3] = st #action
                    int_obs[:,m,4] = torch.ones(batch_size) * self.couplings[l] #coupling
                    m = m + 1
                l = l + 1
            
        return x, Q, logJ, int_obs

	# regular call up to 2*tb-layer with nograd
    def up_to_block_nograd(self, x, s0, tb):
        with torch.no_grad():
            x, Q, logJ = self.forward_up_to_layer(x, 2*tb)
        x, dQ, dlogJ = self.layers[2*tb].forward(x) #forward only on the smearing layer of the target block!
        Q += dQ  #unused for now
        logJ += dlogJ
        st = self.action(x, self.couplings[tb])
        w = st - s0 - Q - 2.0*logJ
        return x, w, st, Q, 2.0*logJ

	#regular forward up to tl-th layer
    def forward_up_to_layer(self, x, tl):
        Q = torch.zeros(x.shape[0])
        logJ = torch.zeros(x.shape[0])
        for layer in self.layers[0:tl]:
            x, dQ, dlogJ = layer.forward(x)
            Q += dQ
            logJ += dlogJ
        return x, Q, logJ

    def compute_metrics(self, w):
        wd = w.detach()
        wdm = wd.mean()
        expwm = torch.mean(torch.exp(-(wd-wdm)))
        DF = - torch.log(expwm) + wdm
        ess = (expwm**2)/torch.mean(torch.exp(-2.0*(wd-wdm)))
        return DF, ess

    def sample_(self, x, s0):
        with torch.no_grad():
            return self(x, s0)

    def print_parameters(self, file):
        b = 0
        for layer in self.layers:
            if (b%2 == 0):
                with open(file, 'a') as ff:
                    rr = layer.rho_root
                    for sm in range(layer.smearing_steps):
                        for m in range(8):
                            #ff.write(str(torch.mm(rr[sm,m,:,:]**2, rr[sm,m,:,:].T**2)) + '\n')
                            par = grab(rr[sm,m,:]**2)
                            ff.write(str(b//2) + ' ' + str(sm) + ' ' + str(m) + ' ')
                            if len(par) == 1:
                                ff.write(str(par[0]) + '\n')
                            else:
                                for d in range(4):
                                    ff.write(str(par[d]) + ' ')
                                ff.write(str(par.sum()/3) + '\n')
            b += 1


def make_NED(betas, action, update, device, nupdates=1):
    layers=[]
    for b in betas:
        l = NED_update(action, b, update, device, nupdates=nupdates)
        layers.append(l)
    flow = Flow(action, betas, torch.nn.ModuleList(layers).to(device))
    return flow


def make_SNF(betas, action, update, N, mask, D, jac_shape, scale_fac, rho_shape, rho_root, device, smearing_steps=1):
    layers = []
    b = 0
    for beta in betas:
        smr = Smearing(N, mask, D, jac_shape, scale_fac, rho_shape, device, rho_root[b], smearing_steps)
        layers.append(smr)
        stb = NED_update(action, beta, update, device)
        layers.append(stb)
        b += 1
    flow = Flow(action, betas, torch.nn.ModuleList(layers).to(device))
    return flow
