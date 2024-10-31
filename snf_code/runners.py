import numpy as np
import time
import torch
import matplotlib.cbook as cbook
from pathlib import Path
from snf_code.utils import save, print_metrics, write, grab, get_lr, print_logfile

def train(flow, prior, optimizer, scheduler, N_era, N_epoch, info, root, weights_path, device):

    history_file = root + '_history.log'
    with open(history_file, 'a') as f:
        f.write(info)
        f.write('N_era=' + str(N_era) + '\n'+'N_epoch=' + str(N_epoch) + '\n'+'batch_size=' + str(prior.init_shape[0]) + '\n'+ 'base_lr=' + str(get_lr(optimizer)) + '\n')
    
    history = {'loss':[],'ESS':[],'dF':[],'var_loss':[], 'heat':[], 'logJ':[]}#, 'acceptance':[]}
    t0 = time.time()

    x_save, _ = prior()
    x_save = x_save.detach()
    for era in range(N_era):
        for epoch in range(N_epoch):
            optimizer.zero_grad()
            x, s0 = prior(cfgs=x_save.to(device))
            x = x.detach()
            s0 = s0.detach()
            x_save = x.clone().cpu()
            #x=x.requires_grad_()
            #s=s.requires_grad_()
            x, w, _, q, logJ, _ = flow(x, s0)
            df, ess = flow.compute_metrics(w)
      
            loss = w.mean()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            history['loss'].append(grab(loss))
            history['ESS'].append(grab(ess))
            history['dF'].append(grab(df))
            history['var_loss'].append(grab(w.var()))
            history['heat'].append(grab(q))
            history['logJ'].append(grab(logJ))
            #history['acceptance'].append(grab(acc.mean())) 

            print_metrics(history_file, history, era, epoch, N_epoch, t0)

    save(flow.layers, [optimizer], weights_path)
    write(history, root)

def train_craft1(flow, prior, optimizers, schedulers, N_era, N_epoch, info, root, weights_path, device):
    
    history_file = root + '_history.log'
    with open(history_file, 'a') as f:
        f.write(info)
        f.write('Training block by block separately (CRAFT1) \n')
        f.write('N_era=' + str(N_era) + '\n'+'N_epoch=' + str(N_epoch) + '\n'+'batch_size=' + str(prior.init_shape[0]) + '\n'+ 'base_lr=' + str(get_lr(optimizers[0])) + '\n')
    
    #, 'acceptance':[]}
    t0 = time.time()

    for block in range(len(flow.couplings)):
        history = {'loss':[], 'ESS':[], 'dF':[], 'var_loss':[], 'heat':[], 'logJ':[]}

        x_save, _ = prior()
        x_save = x_save.detach()
        for era in range(N_era):
            for epoch in range(N_epoch):
                optimizers[block].zero_grad()
                x, s0 = prior(cfgs=x_save.to(device))
                x = x.detach()
                s0 = s0.detach()
                x_save = x.clone().cpu()

                x, w, q, logJ = flow.up_to_block_nograd(x, s0, block)
                df, ess = flow.compute_metrics(w)
                
                loss = w.mean()
                #print(loss.requires_grad)
                loss.backward()
                optimizers[block].step()
                schedulers[block].step(loss)

                history['loss'].append(grab(loss))
                history['ESS'].append(grab(ess))
                history['dF'].append(grab(df))
                history['var_loss'].append(grab(w.var()))
                history['heat'].append(grab(q))
                history['logJ'].append(grab(logJ))
                #history['acceptance'].append(grab(acc.mean())) 

                print_metrics(history_file, history, era, epoch, N_epoch, t0)

    save(flow.layers, optimizers, weights_path) #only the optimizer of the last block is saved!
    write(history, root)

def train_craft2(flow, prior, optimizers, schedulers, N_era, N_epoch, info, root, weights_path, device):

    history_file = root + '_history.log'
    with open(history_file, 'a') as f:
        f.write(info)
        f.write('Training block by block in one go (CRAFT2) \n')
        f.write('N_era=' + str(N_era) + '\n'+'N_epoch=' + str(N_epoch) + '\n'+'batch_size=' + str(prior.init_shape[0]) + '\n'+ 'base_lr=' + str(get_lr(optimizers[0])) + '\n')
    
    history = {'loss':[],'ESS':[],'dF':[],'var_loss':[], 'heat':[], 'logJ':[]}#, 'acceptance':[]}
    t0 = time.time()

    x_save, _ = prior()
    x_save = x_save.detach()
    for era in range(N_era):
        for epoch in range(N_epoch):
            x, s0 = prior(cfgs=x_save.to(device))
            x = x.detach()
            s0 = s0.detach()
            x_save = x.clone().cpu()
            #x=x.requires_grad_()
            #s=s.requires_grad_()

            q = torch.zeros(x.shape[0])
            logJ = torch.zeros(x.shape[0])
            b = 0
            for block in range(len(flow.couplings)):
                x, dq, dlogJ = flow.layers[2*block].forward(x) #even ones, only smearing
                q += dq #unused(zero)
                logJ += dlogJ
                st = flow.action(x, flow.couplings[block])
                w = st - s0 - q - 2.0*logJ
                loss = w.mean()
                loss.backward()
                optimizers[block].step()
                schedulers[block].step(loss)
                optimizers[block].zero_grad()
                x = x.detach()
                q = q.detach()
                logJ = logJ.detach()
                x, dq, dlogJ = flow.layers[2*block+1].forward(x) #odd ones, only MC
                q += dq
                logJ += dlogJ #unused(zero)

            
            st = flow.action(x, flow.couplings[-1])
            w = st - s0 - q - 2.0*logJ
            loss = w.mean()
            df, ess = flow.compute_metrics(w)
            logJ = 2.0*logJ

            history['loss'].append(grab(loss))
            history['ESS'].append(grab(ess))
            history['dF'].append(grab(df))
            history['var_loss'].append(grab(w.var()))
            history['heat'].append(grab(q))
            history['logJ'].append(grab(logJ))
            #history['acceptance'].append(grab(acc.mean())) 

            print_metrics(history_file, history, era, epoch, N_epoch, t0)

    save(flow.layers, optimizers, weights_path)
    write(history, root)

def eval(flow, prior, nmeas, base_file, info, details, int_meas=1):

    log_file = base_file + '.log'
    dat_file = base_file + '.dat'
    work_file = base_file + '_work.dat'
    heat_file = base_file + '_heat.dat'
    action_file = base_file + '_action.dat'

    ##############################
    #Sampling
    t0 = time.time()
  
    with torch.no_grad(): 
        x0, s0 = prior()

        batch_size = x0.shape[0]

        with open(log_file, 'a') as f:
            f.write(info)
            f.write('nmeas=' + str(nmeas) + ' batch_size=' + str(batch_size) + ' intermediate_measures=' + str(int_meas) + '\n')

        obs = []
        for i in range(nmeas):
            x0, s0 = prior(cfgs=x0)
            _, _, _, _, _, int_obs = flow(x0, s0, int_meas)
            obs.append(int_obs)

    t1 = time.time()
    ##############################

    with open(log_file, 'a') as f:
        f.write('Sampling time: ' + str(t1-t0) + '\n')

    for im in range(int_meas):
        work = []
        work2 = []
        expw = []
        expw2 = []
        heat = []
        action = []
        actionrw = []
        runs = []

        for bs in range(batch_size):
            work.append(np.zeros(nmeas))
            work2.append(np.zeros(nmeas))
            expw.append(np.zeros(nmeas))
            expw2.append(np.zeros(nmeas))
            heat.append(np.zeros(nmeas))
            action.append(np.zeros(nmeas))
            actionrw.append(np.zeros(nmeas))
            runs.append('run|' + str(bs))

        for i in range(nmeas):
            w = grab(obs[i][:,im,0])
            q = grab(obs[i][:,im,1])
            sf = grab(obs[i][:,im,3])
            cplng = grab(obs[i][0,im,4])

            for bs in range(batch_size):
                work[bs][i] = w[bs]
                work2[bs][i] = w[bs]**2
                expw[bs][i] = np.exp(-(w[bs]-work[0][0]))
                expw2[bs][i] = np.exp(-2.0*(w[bs]-work[0][0]))
                heat[bs][i] = q[bs]
                action[bs][i] = sf[bs]
                actionrw[bs][i] = sf[bs]*np.exp(-(w[bs]-work[0][0]))

        with open(work_file, 'a') as ff:
            for bs in range(batch_size):
                for i in range(nmeas):
                    ff.write("%f " % np.mean(work[bs][i]))
                ff.write("\n")

        with open(heat_file, 'a') as ff:
            for bs in range(batch_size):
                for i in range(nmeas):
                    ff.write("%f " % np.mean(heat[bs][i]))
                ff.write("\n")

        with open(action_file, 'a') as ff:
            for bs in range(batch_size):
                for i in range(nmeas):
                    ff.write("%f " % np.mean(action[bs][i]))
                ff.write("\n")

        details = details + ' ' + str(cplng) + ' ' + str(im+1)
        print_logfile(work, work2, expw, expw2, heat, action, actionrw, runs, details, log_file, dat_file)


def repeat_analysis(base_file, info, details, int_meas=1):

    log_file = base_file + '.log'
    dat_file = base_file + '.dat'
    work_file = base_file + '_work.dat'
    heat_file = base_file + '_heat.dat'
    action_file = base_file + '_action.dat'

    with cbook.get_sample_data(work_file) as file:
        work_array = np.loadtxt(work_file)

    bsxim = work_array.shape[0]
    nmeas = work_array.shape[1]
    batch_size = bsxim // int_meas

    if Path(heat_file).is_file():
        with cbook.get_sample_data(heat_file) as file:
            heat_array = np.loadtxt(heat_file)
    else:
        heat_array = np.zeros((bsxim, nmeas))

    if Path(action_file).is_file():
        with cbook.get_sample_data(action_file) as file:
            action_array = np.loadtxt(action_file)
    else:
        action_array = np.zeros((bsxim, nmeas))

    with open(log_file, 'a') as f:
        f.write(info)
        f.write('nmeas=' + str(nmeas) + ' batch_size=' + str(batch_size) + ' intermediate_measures=' + str(int_meas) + '\n')

    for im in range(int_meas):
        work = []
        work2 = []
        expw = []
        expw2 = []
        heat = []
        action = []
        actionrw = []
        runs = []

        for bs in range(batch_size):
            work.append(work_array[im*int_meas+bs,:])
            work2.append(work_array[im*int_meas+bs,:]**2)
            expw.append(np.exp(-(work_array[im*int_meas+bs,:]-work_array[im*int_meas,0])))
            expw2.append(np.exp(-2.0*(work_array[im*int_meas+bs,:]-work_array[im*int_meas,0])))
            heat.append(heat_array[im*int_meas+bs,:])
            action.append(action_array[im*int_meas+bs,:])
            actionrw.append(action_array[im*int_meas+bs,:]*np.exp(-(work_array[im*int_meas+bs,:]-work_array[im*int_meas,0])))
            runs.append('run|' + str(bs))

        details += ' ' + str(nmeas) + ' ' + str(batch_size) + ' ' + str(cplng) + ' ' + str(im+1)
        print_logfile(work, work2, expw, expw2, heat, action, actionrw, runs, details, log_file, dat_file)