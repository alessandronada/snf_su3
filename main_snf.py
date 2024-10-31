import torch
import argparse
import numpy as np
import snf_code.runners as runners
import snf_code.utils as utils
import snf_code.montecarlo as mc
import snf_code.flow as fw
import snf_code.distributions as dst

parser = argparse.ArgumentParser()

#Lattice
parser.add_argument("--D", type=int, default=4)
parser.add_argument("--T", type=int, default=4)
parser.add_argument("--L", type=int, default=4)

#Flow in beta
parser.add_argument("--N_col", type=int, default=3)
parser.add_argument("--beta0", type=float, default=6.0)
parser.add_argument("--beta_steps", type=int, default=4)
parser.add_argument("--beta_tar", type=float, default=6.2)
parser.add_argument("--prot", type=int, default=0)

#Prior 
parser.add_argument("--prior_thsteps", type=int, default=50)
parser.add_argument("--prior_mcsteps", type=int, default=2)

#Heatbath
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--orsteps", type=int, default=1)
parser.add_argument("--updates_per_step", type=int, default=1)

#Smearing
parser.add_argument("--scale_fac", type=float, default=0.01**4)
parser.add_argument("--smearing_steps", type=int, default=1)
parser.add_argument("--isotropic_rho", type=int, default=0)
parser.add_argument("--initialize_rho", type=int, default=0)

#Training
parser.add_argument("--training", action="store_true")
parser.add_argument("--base_lr", type=float, default=0.0001)
parser.add_argument("--N_era", type=int, default=100)
parser.add_argument("--N_epoch", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--patience", type=int, default=500)
parser.add_argument("--min_lr", type=float, default=1e-7)
parser.add_argument("--sched_scale", type=float, default=0.92)
parser.add_argument("--craft", type=int, default=0)

#Measurements
parser.add_argument("--measures", action="store_true")
parser.add_argument("--nmeas", type=int, default=2000)
parser.add_argument("--int_meas", type=int, default=1)
parser.add_argument("--noload", action="store_true")#don't load parameters, use initialization
parser.add_argument("--reanalysis", action="store_true")#just reanalyse saved data

#Transfer
parser.add_argument("--cont_run", type=int, default=0)
parser.add_argument("--transfer", type=int, default=0)
parser.add_argument("--T_transf", type=int, default=4)
parser.add_argument("--L_transf", type=int, default=4)
parser.add_argument("--beta0_transf", type=float, default=6.0)
parser.add_argument("--beta_tar_transf", type=float, default=6.2)

#path
parser.add_argument("--base_root",type=str, default="/leonardo/home/userexternal/anada000")
parser.add_argument("--base_path",type=str, default="snf_sun/test/")


if torch.cuda.is_available():
    device = 'cuda'
    #float_dtype = np.float64 # single
    #torch.set_default_tensor_type(torch.cuda.DoubleTensor)  
    torch.set_default_dtype(torch.float64)
    torch.set_default_device('cuda') 
else:
    device = 'cpu'
    #float_dtype = np.float64 # double
    #torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device('cpu')

args = parser.parse_args()

base_root=args.base_root
base_root+=args.base_path
info='Device '+ device
if device=='cuda':
    info+=' '+torch.cuda.get_device_name(torch.cuda.current_device())

N_col=args.N_col
D=args.D
T=args.T
L=args.L
b0 = args.beta0
bf = args.beta_tar
beta_steps = args.beta_steps

training=args.training
measures=args.measures
batch_size = args.batch_size

#Shapes
init_shape=[batch_size,D,T]
jac_shape=[batch_size,T]
if N_col==3:
    rn_shape=[batch_size,18,D,T]
elif N_col==2:
    rn_shape=[batch_size,6,D,T]
for i in range(D-1):
    init_shape+=[L]
    jac_shape+=[L]
    rn_shape+=[L]
init_shape+=[N_col]
jac_shape+=[N_col*N_col,N_col*N_col]
init_shape=tuple(init_shape)
rn_shape=tuple(rn_shape)
jac_shape=tuple(jac_shape)

base_root += 'T' + str(T) + '_L' + str(L)+'_D'+str(D)
info += '\n'+'D='+str(D-1)+'+1'+' T/a=' + str(T) + ' L/a=' + str(L)+ '\n'

base_root += '_SU' + str(N_col) + '_b0' + str(b0) + '_bf' + str(bf) + '_nbl' + str(beta_steps)
info += 'SU(' + str(N_col) + ') flow from beta0=' + str(b0) + ' to target_beta=' + str(bf) + ' with ' + str(beta_steps) + ' blocks \n'

ar00 = utils.ar0(b0)
ar0t = utils.ar0(bf)

info += 'Prior lattice spacing a0 = ' + str(ar00*0.5) + 'fm \n'
info += 'Target lattice spacing at = ' + str(ar0t*0.5) + 'fm \n'
info += 'Physical size of target lattice L = ' + str(L*ar0t*0.5) + 'fm \n\n'

if args.prot==1:
    latspacings = np.linspace(ar00 + (ar0t - ar00)/beta_steps, ar0t, beta_steps)
    betas = utils.betafromar0(latspacings)
    info += '  (linear protocol in a/r0) \n'
    base_root += '_prt' + str(args.prot)
    print(latspacings)
    print(betas)
elif args.prot==2:
    t = np.linspace(0.1,1,beta_steps)
    gamma = 0.2
    latspacings = gamma*ar0t*t**2 +  ((1.0-gamma) * ar0t - ar00) * t + ar00
    betas = utils.betafromar0(latspacings)
    info += '  (quadratic protocol in a/r0) \n'
    base_root += '_prt' + str(args.prot)
    print(latspacings)
    print(betas)
    #invlatspacings = np.linspace(1/ar0t**2, 1/ar00**2 - (1/ar00**2 - 1/ar0t**2)/beta_steps, beta_steps)
    #betas = np.flip(utils.betafromar0(1/invlatspacings**0.5))
    #info += '  (linear protocol in (r0/a)^2) \n'
    #print(betas)
else:
    betas = torch.linspace(b0 + (bf - b0)/beta_steps, bf, beta_steps)
    info += '  (linear protocol in beta) \n'
    print(betas)

action = dst.S_SUN(D, N_col)
info+= 'Prior: MC chain thermalized at beta0 with ' + str(args.prior_thsteps) + ' steps \n       New configurations sampled from the chain every '+ str(args.prior_mcsteps) +' steps \n' 

#Creating masks, setting update and prior Markov Chain
mask = utils.create_mask(D, T, L).to(device)
update = mc.init_hb(N_col, mask, D, args.orsteps, rn_shape, init_shape)
prior = dst.PriorSUN(init_shape, action, b0, update, args.prior_thsteps, args.prior_mcsteps, device)

scale_fac=args.scale_fac
smearing_steps=args.smearing_steps
isotropic_rho=args.isotropic_rho

if args.stochastic:
    flow = fw.make_NED(betas, action, update, device, nupdates=args.updates_per_step)
    base_root += '_NED_orst' + (str(args.orsteps))
    info += 'NED with one heatbath update and ' + (str(args.orsteps)) + ' overrelaxation steps in each block \n'
    training = False
    measures = True
else:
    if isotropic_rho:
        rho_shape = (smearing_steps, 2*D, 1)
    else:
        rho_shape = (smearing_steps, 2*D, 4)

    base_root += '_SNF_orst' + (str(args.orsteps))
    info += 'SNF with ' + str(smearing_steps) + ' smearing steps, one heatbath update and ' + str(args.orsteps) + ' overrelaxation steps in each block \n'
    if args.initialize_rho:
        rho_root = utils.set_rho(beta_steps, b0, bf, rho_shape)
        base_root += '_rhofit' 
        info += 'Smearing parameters set from fits \n\n'
    else:
        rho_root = []
        for bst in range(beta_steps):
            rho_root.append(None)
        base_root += '_rhoscale' + str(scale_fac) 
        info += 'Scale factor for smearing parameters = ' + str(scale_fac) + ' \n\n'
    
    base_root += '_smst' + str(smearing_steps) + '_craft' + str(args.craft)
        
    flow = fw.make_SNF(betas, action, update, N_col, mask, D, jac_shape, scale_fac, rho_shape, rho_root, device, smearing_steps)

info += 'Path: ' + base_root + '\n'
weights_path = base_root + '.chckpnt'

if training:
    base_lr = args.base_lr
    N_era = args.N_era
    N_epoch = args.N_epoch

    if args.transfer != 0:
       t_weights_path = weights_path.replace("T" + str(T), "T" + str(args.T_transf)).replace("L" + str(L), "L" + str(args.L_transf))
       weights_path = t_weights_path.replace("b0" + str(b0), "b0" + str(args.beta0_transf)).replace("bf" + str(bf), "bf" + str(args.beta_tar_transf))

    if args.craft != 0:
        optimizers = []
        schedulers = []
        for block in range(len(betas)):
            optimizer = torch.optim.Adam(flow.layers[2*block].parameters(), lr=base_lr)
            optimizers.append(optimizer)
            schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.sched_scale, patience=args.patience, min_lr=args.min_lr, verbose=True))
        if args.cont_run != 0:
            utils.load(flow.layers, optimizers, weights_path)
    else:
        optimizer = torch.optim.Adam(flow.layers.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.sched_scale, patience=args.patience, min_lr=args.min_lr, verbose=True)
        if args.cont_run != 0:
            utils.load(flow.layers, optimizer, weights_path)

    if args.craft == 1:
        runners.train_craft1(flow=flow, prior=prior, optimizers=optimizers, schedulers=schedulers, N_era=N_era, N_epoch=N_epoch, info=info, root=base_root, weights_path=weights_path, device=device)
    if args.craft == 2:
        runners.train_craft2(flow=flow, prior=prior, optimizers=optimizers, schedulers=schedulers, N_era=N_era, N_epoch=N_epoch, info=info, root=base_root, weights_path=weights_path, device=device)
    else:
        runners.train(flow=flow, prior=prior, optimizer=optimizer, scheduler=scheduler, N_era=N_era, N_epoch=N_epoch, info=info, root=base_root, weights_path=weights_path, device=device)

    flow.print_parameters(base_root + '.par')

if args.reanalysis:
    base_file = base_root + '_measurement'
    details = str(T) + ' ' + str(L) + ' ' + str(b0) + ' ' + str(bf) + ' ' + str(beta_steps) + ' ' + str(args.prior_mcsteps) + ' ' + str(int_meas)
    runners.repeat_analysis(base_file=base_file, info=info, details=details)
elif measures:
    nmeas = args.nmeas
    int_meas= args.int_meas
    base_file = base_root + '_measurement' 

    if not args.stochastic:
        if not args.noload:
            if args.craft != 0:
                optimizers = []
                utils.load(flow.layers, optimizers, weights_path)
            else:
                optimizer = []
                utils.load(flow.layers, optimizer, weights_path)
    details = str(T) + ' ' + str(L) + ' ' + str(b0) + ' ' + str(bf) + ' ' + str(beta_steps) + ' ' + str(args.prior_mcsteps) + ' ' + str(nmeas) + ' ' + str(int_meas) + ' ' + str(batch_size)
    runners.eval(flow=flow, prior=prior, nmeas=nmeas, base_file=base_file, info=info, details=details, int_meas=int_meas)


