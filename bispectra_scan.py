#!/usr/bin/env python3 
#import
import numpy as np
import dbi_funcs_np
from potential_def_arg import add_bump_reso, add_reso, add_tanh, dbi_IR_quadratic_potential_funcs, quadratic_potential_funcs, starobinsky_pot_funcs

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from numpy.polynomial.legendre import leggauss
from scipy.optimize import curve_fit
import math
from time import time

import config

from evol_deriv import get_spl_index,spline_eval,mode_evolution_cython
import os
import gc
import sys

from orthog_mixed_basis_3d import tidied_eval,corr,print_tetra,set_up_flat_basis,get_coeffs_1d_from_samples,eval_vander,basis_vander,gen_series_product_matrix,gen_exp_coeffs,tidy_coeffs,decomp_zs,construct_and_integrate,reduce_coeffs_augd
import orthog_mixed_basis_3d as ort
from arb_pts_integrator import simps_osc_weights_lg
import quad_sum

from time import time
from func_timeout import func_timeout, FunctionTimedOut
from mpi4py import MPI
#############################################################

#file for scenarios:
scen_file = 'scen_to_fix_2'
#time limit for 1 job
time_max = 600

###########################################################
##define parameters and functions independent to scenarios
l_max = 30-3
pmax = 30

#set the range of k
k_exp_max   = np.log(1.391727e-01) #np.log(1.2*2500./14370.634)
k_exp_min   = -np.log(1000.)+k_exp_max
k_min,k_max = np.exp(k_exp_min),np.exp(k_exp_max)
k_pivot     = 0.05

#set working basis and final basis to decompose the curvature and bispectrum
working_basis = ort.set_up_log_basis(k_min, k_max, l_max, inv=False, verbose=False)
final_basis   = ort.set_up_log_basis(k_min, k_max, l_max+3, inv=True, verbose=False)

# The order of the internal exp expansion.
num_exps_cs = max(65, 2*l_max)

#leggauss
LG_INV_FUNC_RES = 600
LG_LOW_RES = 75
Nk_aug = 2500

# integration_method
integ_method ='lg'

#def functions
def spline_arrs(xs,ys):
    '''Setting up splines'''
    yl = ys[:-1]
    yh = ys[1:]
    xl = np.zeros_like(yl).T
    xh = np.zeros_like(yh).T
    xl[:] = xs[:-1]
    xh[:] = xs[1:]
    xl,xh = xl.T,xh.T
    A = (yl*xh-yh*xl)/(xh-xl)
    B = (yh-yl)/(xh-xl)
    return A,B

def bar(x,k_min,k_max):
    '''Map to [-1,1]'''
    return (2*x-(k_min+k_max))/(k_max-k_min)

def unbar(x,k_min,k_max):
    '''Map to [kmin,kmax]'''
    return 0.5*((k_max-k_min)*x+k_min+k_max)

#decompose zs
def decomp_zs(ks, lg_ws, t_zs, zs, dzs, bkgd_interps, basis_funcs, l_max, late_t_switch):
    t0 = time()
    final_zetas = zs[-1]*np.exp(-1j*ks*bkgd_interps[9](t_zs[-1]))
    ## # Memory problems!
    for i,f in enumerate(zs):
        zs[i] = zs[i].conj()
    #zs[:] = zs[:].conj()
    zs *= final_zetas
    for i,df in enumerate(dzs):
        dzs[i] = dzs[i].conj()
    #dzs[:] = dzs[:].conj()
    dzs *= final_zetas

    kbars = bar(ks,k_min,k_max)
    I_coeffs = np.zeros((len(t_zs),l_max))*1j
    J_coeffs = np.zeros((len(t_zs),l_max))*1j
    fit_check = np.zeros(len(t_zs)//10+1)*1j
    dfit_check = np.zeros(len(t_zs)//10+1)*1j

    mixed_vander = basis_vander(basis_funcs, k_min, k_max,kbars,lg_ws)
    test_vander = eval_vander(basis_funcs, k_min, k_max,kbars)
    tau_s_array = bkgd_interps[9](t_zs)
    t2 = time()
    for i,tau_s in enumerate(tau_s_array):
        if late_t_switch<t_zs[i]:
            zs[i] *= np.exp(1j*ks*tau_s)
            dzs[i] *= np.exp(1j*ks*tau_s)
        else:
            i_switch = i

        if integ_method=='lg':
            I_coeffs[i] = get_coeffs_1d_from_samples(zs[i],mixed_vander)
            J_coeffs[i] = get_coeffs_1d_from_samples(dzs[i]/ks,mixed_vander)
        if i%10==0:
            fg = np.dot(np.dot(test_vander.T,I_coeffs[i]),zs[i])
            gg = np.dot(zs[i],zs[i])
            fit_check[i//10] = fg/gg
            fg = np.dot(np.dot(test_vander.T,J_coeffs[i]),dzs[i]/ks)
            gg = np.dot(dzs[i]/ks,dzs[i]/ks)
            dfit_check[i//10] = fg/gg
    
    I_coeffs_final = np.zeros( (l_max,len(t_zs)) ).T*1j
    J_coeffs_final = np.zeros( (l_max,len(t_zs)) ).T*1j

    in_basisA = set_up_flat_basis(k_min,k_max,num_exps_cs)
    in_basisB = np.copy(basis_funcs)

    prod = gen_series_product_matrix(in_basisA,in_basisB,basis_funcs, k_min, k_max)

    exp_fit_check = np.zeros(len(t_zs)//10+1)*1j
    dexp_fit_check = np.zeros(len(t_zs)//10+1)*1j
    test_vander = eval_vander(basis_funcs, k_min, k_max, kbars)
    
    decomp_xs,decomp_ws = leggauss(LG_INV_FUNC_RES)
    zs = basis_funcs[0](decomp_xs)
    norm_0 = np.dot(zs**2,decomp_ws)
    rotate_inds = np.where(late_t_switch>t_zs)[0]
    non_rotate_inds = np.where(late_t_switch<=t_zs)[0]

    exp_cs = gen_exp_coeffs(k_min,k_max,tau_s_array[rotate_inds],num_exps_cs,norm_0)
  
    batch_size = 100
    batch_breaks = np.arange(0,len(rotate_inds),batch_size)[1:]

    batches = np.split(rotate_inds,batch_breaks)

    for batch in batches:
        I_coeffs_final[batch, :] = np.einsum('pi,pj,jik->pk', I_coeffs[batch], exp_cs[batch], prod, optimize=True)
        J_coeffs_final[batch, :] = np.einsum('pi,pj,jik->pk', J_coeffs[batch], exp_cs[batch], prod, optimize=True)

    I_coeffs_final[non_rotate_inds,:] = I_coeffs[non_rotate_inds]
    J_coeffs_final[non_rotate_inds,:] = J_coeffs[non_rotate_inds]

    len_check = min(len(t_zs[::10]),len(fit_check.real))
    #to_print = np.array([t_zs[::10][:len_check],fit_check.real[:len_check],fit_check.imag[:len_check],dfit_check.real[:len_check],dfit_check.imag[:len_check],exp_fit_check.real[:len_check],exp_fit_check.imag[:len_check],dexp_fit_check.real[:len_check],dexp_fit_check.imag[:len_check]]).T
    return I_coeffs_final.T,J_coeffs_final.T

#integrate and tidy up coefficients of bispectrum
def construct_and_integrate(t_zs, I_coeffs, J_coeffs, bkgd_interps, basis_funcs, l_max, final_basis, N_start_integ, beta_activation, beta_margin):

    eps_array     = bkgd_interps[1](t_zs)
    eps_s_array   = bkgd_interps[2](t_zs)
    eta_array     = bkgd_interps[3](t_zs)
    phi_array     = bkgd_interps[6](t_zs)
    c_s_array     = bkgd_interps[7](t_zs)
    H_array       = bkgd_interps[8](t_zs)
    tau_s_array   = bkgd_interps[9](t_zs) 
    cheat = np.ones(len(t_zs))*(1+0j)
    cheat[t_zs<N_start_integ] *= np.exp(-beta_activation*np.abs(tau_s_array[t_zs < N_start_integ]-tau_s_array[t_zs < N_start_integ][-1])**2)
    cheat[t_zs< N_start_integ- beta_margin] *= 0
    tau_s_simps_weights = simps_osc_weights_lg(tau_s_array,3*k_max)*np.exp(t_zs)/c_s_array

    cheat *= tau_s_simps_weights*np.exp(-3j*k_max*tau_s_array)
    sys.stdout.flush()
    bkgd_coeffs_0 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,0,phi_array)*cheat
    bkgd_coeffs_1 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,1,phi_array)*cheat
    bkgd_coeffs_2 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,2,phi_array)*cheat
    bkgd_coeffs_3 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,3,phi_array)*cheat
    bkgd_coeffs_4 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,4,phi_array)*cheat
    bkgd_coeffs_5 = config.bkgd_coeff(t_zs,H_array,eps_array,eta_array,c_s_array,eps_s_array,5,phi_array)*cheat
    bkgd_coeffs_list = [bkgd_coeffs_0,bkgd_coeffs_1,bkgd_coeffs_2,bkgd_coeffs_3,bkgd_coeffs_4,bkgd_coeffs_5]
    IJ_list = [[1,1,1],[1,1,0],[0,0,0],[1,1,0],[1,1,0],[0,0,1]]

    coeffs = np.zeros((l_max,l_max,l_max))

    assert len(bkgd_coeffs_0)==np.shape(I_coeffs)[1]
    assert len(bkgd_coeffs_1)==np.shape(J_coeffs)[1]
    maptimer = np.zeros(2)
    intgd_smpls = 10
    to_print_labels = ['t','tau']
    early_print_inds = np.where((t_zs>N_start_integ-0.1)*(t_zs < N_start_integ+0.1))[0]#[::intgd_smpls]
    to_print_slice = slice(max(-len(t_zs),-3000*intgd_smpls),None,intgd_smpls)
    late_print_inds = np.arange(len(t_zs))[to_print_slice]
    full_print_inds = np.array(list(early_print_inds)+list(late_print_inds),dtype=int)
    to_print = [t_zs[full_print_inds],tau_s_array[full_print_inds]]

    for s in range(len(bkgd_coeffs_list)):
        for i in range(len(basis_funcs)):
            temp = np.copy(bkgd_coeffs_list[s])[full_print_inds]
            if IJ_list[s][0]==0:
                temp *= I_coeffs[i,:][full_print_inds]
            else:
                temp *= J_coeffs[i,:][full_print_inds]
            if IJ_list[s][1]==0:
                temp *= I_coeffs[i,:][full_print_inds]
            else:
                temp *= J_coeffs[i,:][full_print_inds]
            if IJ_list[s][2]==0:
                temp *= I_coeffs[i,:][full_print_inds]
            else:
                temp *= J_coeffs[i,:][full_print_inds]
            temp *= np.exp(3j*k_max*tau_s_array[full_print_inds]*(t_zs[full_print_inds]>N_start_integ+1))
            temp /= tau_s_simps_weights[full_print_inds]
            to_print.append( np.copy(temp.imag) )
            #to_print_labels.append(f'{s}-{i}-{i}-{i}')
            to_print_labels.append(str(s)+'-'+str(i)+'-'+str(i)+'-'+str(i))
   
    coeff_results = np.zeros((len(config.shape_indices),l_max,l_max,l_max))
    temp1I = np.zeros(len(bkgd_coeffs_0))*1.j
    temp1J = np.zeros(len(bkgd_coeffs_0))*1.j
    temp2 = np.zeros(len(bkgd_coeffs_0))*1.j
    final_part = np.zeros_like(I_coeffs)*1.j
    #for raw_ind, bkgd_vals in enumerate(bkgd_coeffs_list):
        #print('# Sum of bkgd', raw_ind, '=', np.sum(bkgd_vals), flush=True) 
    quad_sum.do_the_integrals(I_coeffs, J_coeffs, coeff_results, config.shape_indices, bkgd_coeffs_0, bkgd_coeffs_1, bkgd_coeffs_2, bkgd_coeffs_3, bkgd_coeffs_4, bkgd_coeffs_5, temp1I, temp1J, temp2, final_part)

    coeffs = coeff_results.reshape(len(config.shape_indices),l_max,l_max,l_max)
    raw_coeffs = np.copy(coeffs)
    #for raw_ind, raw_cs in enumerate(raw_coeffs):
        #print('# Sum of raw', config.shape_indices[raw_ind], '=', np.sum(raw_cs), flush=True)

    for i_s in range(len(config.shape_indices)):
        for j in range(l_max):
            for i in range(j):
                coeffs[i_s,i,j] = coeffs[i_s,j,i]

    tidied_coeffs, basis_funcs_padded = tidy_coeffs(coeffs, basis_funcs, k_min, k_max, final_basis)

    return tidied_coeffs, basis_funcs_padded

#estimate fnl and std
##load sigma matrix
sigma_2001_l30 = np.load("sigma_2001_l30.npy")
#load Gamma_XXX matrix
Gamma_2001_TTT = np.fromfile("gamma_DX12_v3_smica_TTT_2-2000_4_9_2001", dtype="d").reshape((2001,2001))
Gamma_2001_TTE = np.fromfile("gamma_DX12_v3_smica_TTE_2-2000_2-1500_4_9_2001", dtype="d").reshape((2001,2001))
Gamma_2001_TEE = np.fromfile("gamma_DX12_v3_smica_TEE_2-2000_2-1500_4_9_2001", dtype="d").reshape((2001,2001))
Gamma_2001_EEE = np.fromfile("gamma_DX12_v3_smica_EEE_2-1500_4_9_2001", dtype="d").reshape((2001,2001))
##load lambda_XXX_inv
lambda_TTT_inv = np.fromfile("orthol_TTT_2-2000_9_2001", dtype="d").reshape((2001,2001))
lambda_TTE_inv = np.fromfile("orthol_TTE_2-2000_2-1500_9_2001", dtype="d").reshape((2001,2001))
lambda_TEE_inv = np.fromfile("orthol_TEE_2-2000_2-1500_9_2001", dtype="d").reshape((2001,2001))
lambda_EEE_inv = np.fromfile("orthol_EEE_2-1500_9_2001", dtype="d").reshape((2001,2001))
##load beta, substract ISW and Lensing
beta = np.loadtxt('beta_DX12_smica_case1_HP_T+E_2-2000_2-1500_9_2001.unf').reshape(8004) 
modeISW_TTT = np.dot(np.fromfile("modes_DX12_v3_smica_TTT_2-2000_9_2001_999_0_0", dtype="d"), lambda_TTT_inv)
modePS_TTT = np.dot(np.fromfile("modes_DX12_v3_smica_TTT_2-2000_9_2001_998_0_0", dtype="d"), lambda_TTT_inv)
#define fsky factor
fskyT = 0.78971
fskyE = 0.78036
fskyTTT = fskyT
fskyTTE = (fskyT*fskyT*fskyE)**(1/3)
fskyTEE = (fskyT*fskyE*fskyE)**(1/3)
fskyEEE = fskyE
#remove ISW and PS effect from data
r1 = np.dot(modePS_TTT, modePS_TTT)
r1 = fskyTTT*r1/6
x1 = 0
for i in range(2001):
    x1 += modePS_TTT[i]*(beta[i]-fskyTTT*modeISW_TTT[i])
x1 = x1/(6e0*r1)
beta_clean = np.copy(beta)
for i in range(2001):
    beta_clean[i] = beta[i] - fskyTTT*(modeISW_TTT[i] + x1*modePS_TTT[i])

beta_TTT = np.zeros(2001)
beta_TTE = np.zeros(2001)
beta_TEE = np.zeros(2001)
beta_EEE = np.zeros(2001)
for i in range(2001):
    beta_TTT[i] = beta_clean[i]
    beta_TTE[i] = beta_clean[i+2001]
    beta_TEE[i] = beta_clean[i+4002]
    beta_EEE[i] = beta_clean[i+6003]
##load covbeta
covbeta = np.loadtxt("covariance_DX12_smica_case1_HP_T+E_2-2000_2-1500_9_2001.unf").reshape((8004,8004))

def fnl_estimator(coeffs, pmax):
    ##step 1: convert coeffs from primodal to planck primordial basis with sigma (=gamma_inv*omega_2d)
    alpha = coeffs.reshape(pmax**3)
    alpha_bar = np.dot(sigma_2001_l30, alpha) #alpha_bar:coeffs w.r.t Planck primordial basis
    #remove the Planck power spectrum amplitude
    norm_factor = 6*3/5*(2.1e-9*(2*np.pi**2))**2
    alpha_bar = alpha_bar/norm_factor

    ##step 2: convert to CMB basis with Gamma_XXXs, and rotate to orthonormal basis R with lambda_XXX_inv
    alpha_TTT = np.dot(Gamma_2001_TTT, alpha_bar)
    alpha_TTE = np.dot(Gamma_2001_TTE, alpha_bar)
    alpha_TEE = np.dot(Gamma_2001_TEE, alpha_bar)
    alpha_EEE = np.dot(Gamma_2001_EEE, alpha_bar)

    alpha_RTTT = np.dot(alpha_TTT, lambda_TTT_inv)
    alpha_RTTE = np.dot(alpha_TTE, lambda_TTE_inv)
    alpha_RTEE = np.dot(alpha_TEE, lambda_TEE_inv)
    alpha_REEE = np.dot(alpha_EEE, lambda_EEE_inv)
    
    ##step 3: estimate fnl and std from beta and covbeta
    esti = np.dot(beta_TTT, alpha_RTTT)+3*np.dot(beta_TTE, alpha_RTTE)+3*np.dot(beta_TEE, alpha_RTEE)+np.dot(beta_EEE, alpha_REEE)
    norm = fskyTTT*np.dot(alpha_RTTT, alpha_RTTT)+3*fskyTTE*np.dot(alpha_RTTE, alpha_RTTE)+3*fskyTEE*np.dot(alpha_RTEE, alpha_RTEE)+fskyEEE*np.dot(alpha_REEE, alpha_REEE)
    deno = np.dot(alpha_RTTT, alpha_RTTT)+3*np.dot(alpha_RTTE, alpha_RTTE)+3*np.dot(alpha_RTEE, alpha_RTEE)+np.dot(alpha_REEE, alpha_REEE)
    fnl = esti/norm
    
    stdev = 0
    for i in range(2001):
        for j in range(2001):
            stdev+= alpha_RTTT[i]*covbeta[i][j]*alpha_RTTT[j]

    for i in range(2001):
        for j in range(2001):
            stdev+= 9*alpha_RTTE[i]*covbeta[i+2001][j+2001]*alpha_RTTE[j]

    for i in range(2001):
        for j in range(2001):
            stdev+= 9*alpha_RTEE[i]*covbeta[i+2*2001][j+2*2001]*alpha_RTEE[j]   

    for i in range(2001):
        for j in range(2001):
            stdev+= alpha_REEE[i]*covbeta[i+3*2001][j+3*2001]*alpha_REEE[j]  

    fnl_std = np.sqrt(stdev)/norm

    return fnl, fnl_std

#use MPI to evaluate As and ns for each scenario
def bispectra_scan_ms(scen_arr): #scen_arr: [lambda, V0, beta, phi0, del_N]
    #initialize mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #master rank (rank = 0): allocate tasks and integrate result.
    if rank==0:
        #set the empty array to restore the calculation result of omega matrix element
        result_array = np.zeros((len(scen_arr), 5))
        
         # define list of tasks
        tasklist = (i for i in range(len(scen_arr)))
        #set count of finished ranks
        finished = 0
        #create status object for MPI_probe
        status = MPI.Status()

        for i in range(1,size):
        # check there are enough jobs for the number of ranks
        # -1 is a flag to say "no more work" 
            try:
                message = next(tasklist)
            except StopIteration:
                message = -1
        
            # now we send initial job ID's to slaves
            print("[{}] Sending task: {} to rank {}".format(rank,message,i))
            comm.isend(message, dest=i, tag=i)

        # Now we check for messages from complete jobs then allocate new jobs to the slaves
        finish_task = 0
        while finished<size-1:
            # Check for waiting messages with probe 
            flag = comm.iprobe(status=status, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        
            # if a message is waiting then enter this loop to recieve it
            if flag==True:
                # status object stores the origin of the message (and tag etc..)
                source = status.source
                #recv the message
                test = comm.irecv(source=source, tag=source)
                reply = test.wait()
                #print("[{}] Recieving result: {} from rank {}".format(rank,reply,source))
            
                # now check the reply, -1 means the slave receieved a -1 and it's letting
                # us know that it's finished so we add it to our finshed count
                # otherwise we send it its next job ID (which may be -1 if there are 
                # no more jobs)
                if reply[0]==-1:
                    finished +=1
                    print("[{}] Recieved termination finished count: {}".format(rank,finished))
                else:
                    #print("[{}] Done with result {}".format(rank,reply))
                    result_array[int(reply[5])][0] = reply[0]
                    result_array[int(reply[5])][1] = reply[1]
                    result_array[int(reply[5])][2] = reply[2]
                    result_array[int(reply[5])][3] = reply[3]
                    result_array[int(reply[5])][4] = reply[4]

                    #print temperoral results
                    finish_task += 1
                    if np.mod(finish_task, 1) == 0:
                        np.savetxt('bispectrum_scan_result_fix_temp_2', result_array)

                    try:
                        message = next(tasklist)
                    except StopIteration:
                        message = -1
                    print("[{}] Sending task: {} to rank {}".format(rank,message,source))
                    comm.isend(message, dest=source, tag=source)
        
    # Slave ranks section
    else:
        result_array = np.zeros((len(scen_arr), 2))
        # this loop keeps us checking for jobs until we recieve a -1
        while True:
            # recvout next job ID
            test = comm.irecv(source=0, tag=rank)
            task = test.wait()#task: job ID
            #print("[{}] Recieving task: {} from rank {}".format(rank,task,0))
        
            # is job ID = -1 then no more jobs and we can stop
            # We let the master know we are stopping by sending -1 back
            # Otherwise we do the job associated with the ID we recieve
            if task==-1:
                comm.isend(np.array([-1]), dest=0, tag=rank)
                print("[{}] Sending termination to rank {}".format(rank,0))
                break
            else:
                lambda_dbi = scen_arr[task][0]
                V0 = scen_arr[task][1]
                beta_dbi_IR = scen_arr[task][2]
                phi_0 = scen_arr[task][3]
                del_N_cross = scen_arr[task][4]#the number of e-folds before the end of inflation that the mode with pivot scale crossed the horizon

                t_init = 0 # determined while delta_N and kpivot fixed. Set this = 0 here.
                t_final = 18. #fixed

                #calculate initial conditions
                f,f_1,f_11 = dbi_funcs_np.warp_funcs(lambda_dbi)
                V,V_1,V_11 = dbi_IR_quadratic_potential_funcs(np.sqrt(beta_dbi_IR*V0/3.),V0)

                c_s_0 = math.sqrt(1./(1+f(phi_0)*V_1(phi_0)**2/(3.*V(phi_0))))
                H_0 = math.sqrt((1./(f(phi_0)*c_s_0)+V(phi_0)-1./f(phi_0))/3.)
                phi_dash_0 = -1*-math.sqrt((1-c_s_0**2)/(H_0**2*f(phi_0)))

                #sound speed
                sound_speed,d_log_sound_speed,dd_log_sound_speed = dbi_funcs_np.dbi_c_s_funcs(f,f_1,f_11,V,V_1,V_11) 
                phi_11,phi_111,eps_eta = dbi_funcs_np.eom_funcs(f,f_1,f_11,V,V_1,V_11,sound_speed,d_log_sound_speed,dd_log_sound_speed)

                #The number of modes to evolve.
                Nk = 550//2

                #The steepness of the tapering.
                beta_activation = (1e-4*3*k_max)**2
                beta_margin = 1 #1

                #set timestep for integration
                pts_per_osc = 12.*3./2.
                N_start_integ   = min(np.log(k_min*c_s_0/H_0)-2.5, 0.0) #for DBI
                dt_pref = 0.5*0.5*1.*2*np.pi*H_0/(3*k_max*pts_per_osc*c_s_0)
                early_del_t = 10*dt_pref*np.exp(N_start_integ*0+2-0.0)
                late_del_t  = 5e-4
                early_del_t = min(early_del_t,late_del_t*10)
                late_t_switch = np.log(np.sqrt(k_min*k_max*c_s_0**2)/H_0)

                #tolerance
                atol = 10**(-12)
                rtol = 10**(-12)
                #or
                c_s_atol_scale = (c_s_0*np.exp(-t_init)/H_0)*(7.9e-6/4.6e-3)*(7e-4)
                bkgd_atols = np.array([1e-10*c_s_atol_scale]+[abs(phi_0)*1e-6]+[abs(phi_dash_0)*1e-6]+[H_0*1e-6])
                bkgd_rtols = np.array([1e-20]*4)
                eps_0 = 0.5*phi_dash_0**2/c_s_0
                atol_scale = (7e-7)*math.sqrt(c_s_0/eps_0)#*100
                zeta_atols = np.array([(1e-12)*atol_scale]*Nk*4)
                zeta_rtols = np.array([1e-6]*Nk*4)

                #define functions to get bkgd evolutions (should find a better place to define them rather than inside the loop)
                def bkgd_evolution(y, t):
                    '''Evolve the background'''
                    c_s = sound_speed(y[1],y[2],y[3])
                    eps = 0.5*pow(y[2],2)/c_s
                    dy = np.zeros(4)
                    dy[0] = math.exp(-t)*c_s/y[3]   # tau_s
                    dy[1] = y[2]                    # phi
                    dy[2] = phi_11(y[1],y[2],y[3])  # phi'
                    dy[3] = -eps*y[3]               # H
                    return dy

                #This function will solve the ode and ruturn the value of phi, dphi, H, cs at each time step.
                def get_bkgd_test(t_0, phi_0, t_tot = 10000, phi_e = 100, atol = 10**(-12), rtol = 10**(-12)):  
                    #phi_e: the value of the field we choose to end the inflation.
                    #t_tot: the total number of e-fold that the inflation lasting.
                    del_t = max(dt_pref*np.exp(t_0), early_del_t)
                    del_t = min(del_t, late_del_t*10)# set the initial time step
                    
                    fields = np.zeros(4)
                    fields[0] = 0
                    fields[1] = phi_0
                    fields[2] = phi_dash_0
                    fields[3] = H_0
                    #print(fields)
                    c_s_0 = sound_speed(fields[1],fields[2],fields[3])
                    ddphi_0 = phi_11(fields[1],fields[2],fields[3])
                    eps_s_0 = d_log_sound_speed(fields[1],fields[2],ddphi_0,fields[3])

                    t = t_0
                    t_target = t+del_t

                    bkgd_results = [[t,*fields, V(fields[1]), V_1(fields[1]),c_s_0, eps_s_0, 0]]# [N, tau_s, phi, phi', H, dV,  cs]
                    while (t < t_0 + t_tot)*(fields[1] < phi_e): 
                        #for inflations with phi increase over t, we need to set upper limit of phi to end the inflation
                        tol_scale_dt = np.array([1./del_t,1.,1.,1.])
                        soln = odeint(bkgd_evolution,fields,[t,t_target],atol=atol*tol_scale_dt,rtol=rtol,full_output=0)
                        fields = soln[-1]
                        c_s = sound_speed(fields[1],fields[2],fields[3])
                        ddphi = phi_11(fields[1],fields[2],fields[3])
                        eps_s = d_log_sound_speed(fields[1],fields[2],ddphi,fields[3])
                        epseta = eps_eta(fields[1],fields[2],fields[3])
                        eta   = epseta/(0.5*fields[2]**2/c_s)
                        bkgd_results.append([t_target,*fields,V(fields[1]),V_1(fields[1]),c_s, eps_s, eta])
                        t += del_t
                        del_t = max(dt_pref*np.exp(t), early_del_t)
                        del_t = min(del_t, late_del_t*10)
                        t_target = t + del_t
                    c_s_f = sound_speed(fields[1],fields[2],fields[3])
                    H_f = fields[3]
                    bkgd_results = np.array(bkgd_results)
                    ## # [N,tau_s,phi,phi',H,c_s]
                    bkgd_results[:,1] += -(bkgd_results[:,1][-1] + np.exp(-t)*c_s_f/H_f)
                    bkgd_results = np.transpose(bkgd_results)

                    ## # Setting up bkgd arrays
                    ## # [t,tau_s,phi,phi',H,V,dV,c_s]
                    N_array = bkgd_results[0,:]
                    tau_s_array = bkgd_results[1,:]
                    phi_array = bkgd_results[2,:]
                    dphi_array = bkgd_results[3,:]
                    H_array = bkgd_results[4,:]
                    c_s_array = sound_speed(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
                    eps_array = 0.5*bkgd_results[3,:]**2/c_s_array
                    epseta_array = eps_eta(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
                    eta_array = epseta_array/eps_array
                    ddphi_array = phi_11(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
                    eps_s_array = d_log_sound_speed(bkgd_results[2,:],bkgd_results[3,:],ddphi_array,bkgd_results[4,:])
                    dddphi_array = phi_111(bkgd_results[2,:],bkgd_results[3,:],bkgd_results[4,:])
                    dtau_s_array = c_s_array/(np.exp(bkgd_results[0,:])*bkgd_results[4,:])
                    eps_s_dash_array = dd_log_sound_speed(bkgd_results[2,:],bkgd_results[3,:],ddphi_array,dddphi_array,bkgd_results[4,:],epseta_array)
                    eta_dash_array = (ddphi_array**2/c_s_array + dphi_array*dddphi_array/c_s_array - dphi_array*ddphi_array*eps_s_array/c_s_array - eps_array*eta_array*eps_s_array - eps_array*eta_array**2 - eps_s_dash_array*eps_array)/eps_array
                    P_array = -1.5*eta_array+3*eps_s_array+0.5*epseta_array-0.25*eta_array**2-0.5*eta_dash_array+eps_s_array*eta_array-eps_s_array**2-eps_array*eps_s_array+eps_s_dash_array
                    Q_array = -2-eps_s_array

                    interps_list = [None]*13
                    interp_kind = 'linear'
                    ## # [dtau_s,eps,eps_s,eta,P,Q,phi,c_s,H,tau_s,N_cross(k),N(phi)]
                    interps_list[0] = interp1d(N_array,dtau_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[1] = interp1d(N_array,eps_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[2] = interp1d(N_array,eps_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[3] = interp1d(N_array,eta_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[4] = interp1d(N_array,P_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[5] = interp1d(N_array,Q_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[6] = interp1d(N_array,phi_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[7] = interp1d(N_array,c_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[8] = interp1d(N_array,H_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[9] = interp1d(N_array,tau_s_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    temp_k_cross_array = np.exp(N_array)*H_array/c_s_array
                    interps_list[10] = interp1d(temp_k_cross_array,N_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    interps_list[11] = interp1d(phi_array,N_array,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    evol_func_params = np.array([dtau_s_array,eps_array,eps_s_array,eta_array,P_array,Q_array])
                    interps_list[12] = interp1d(N_array,evol_func_params,bounds_error=False,fill_value='extrapolate',kind=interp_kind)
                    spls = np.array([spline_arrs(N_array,q) for q in evol_func_params])
                    As,Bs = spls[:,0,:],spls[:,1,:]
                    return interps_list,bkgd_results, As, Bs

                #get time steps
                def get_t_zs():
                    '''Get the timesteps'''
                    t = t_init
                    t_target = t+early_del_t
                    del_t = t_target-t
                    t_list = []
                    t_zs = []
                    while t<t_final:
                        t_list.append(t)
                        if t_target>N_start_integ-beta_margin:
                            t_zs += list(np.linspace(t,t_target,10, endpoint=True)[1:])
                        t += del_t
                        del_t = max(dt_pref*np.exp(t),early_del_t)
                        del_t = min(del_t,late_del_t*10)
                        t_target = t + del_t
                    t_zs = np.ascontiguousarray(t_zs)
                    return t_zs,t_list

                def get_zks(kbars,t_zs,t_list,interps_list,As,Bs,Ns):
                    '''Get the zks at each timestep.'''
                    ### [dtau_s,eps,eps_s,eta,P,Q,phi,c_s,H,tau_s]
                    del_t = max(dt_pref*np.exp(t_init), early_del_t)
                    del_t = min(del_t, late_del_t*10)
                    Nk = len(kbars)
                    ks = unbar(kbars,np.exp(k_exp_min), np.exp(k_exp_max))
                    k_min,k_max = np.exp(k_exp_min),np.exp(k_exp_max)

                    fields = np.zeros(4*Nk)

                    ####################################################
                    ## # Set up timesteps
                    t = t_init
                    t_target = t+early_del_t
                    del_t = t_target-t
                    zs = np.zeros((len(t_zs),Nk))*1j
                    dzs = np.zeros((len(t_zs),Nk))*1j

                    changed = np.array([False]*len(ks)) # Must be array!
                    changed_time = np.zeros(len(ks))
                    set_time = np.zeros(len(ks))
                    eps_0 = interps_list[1](t_init)
                    c_s_0 = interps_list[7](t_init)
                    H_0 = interps_list[8](t_init)
                    odeint_time = 0
                    count=0
                    threshold_early = 0
                    pos_re = slice(Nk)
                    pos_im = slice(Nk,2*Nk)
                    vel_re = slice(2*Nk,3*Nk)
                    vel_im = slice(3*Nk,4*Nk)
                    zs_now_temp  = np.zeros((9,Nk)).T*1j
                    dzs_now_temp = np.zeros((9,Nk)).T*1j
                    zs_now = np.zeros_like(zs_now_temp)*1j
                    dzs_now = np.zeros_like(dzs_now_temp)*1j
                    for t in t_list:
                        # If set to 4.5, high k has weird glitch at switch. *Only* high k...?
                        if odeint_time>0:
                            # Should be set by hand to activate before deviations from SR set in.
                            limit_early = math.exp(config.delta_early+t)*H_now/c_s_now
                            if ks[-1]>limit_early:
                                threshold_early = np.where(ks>limit_early)[0][0]
                                if ('tanh' in config.label) and t>8:   #HERE
                                    threshold_early = Nk
                                if ('bump' in config.label) and phi_now<config.phi_f+0.5:
                                    # Yes for tanh, no for reso.
                                    pass
                            else:
                                threshold_early = Nk
                            # Activates once effects from eta,eps_s etc. overtake (kc_s)**2 driving force.
                            limit_late = (math.exp(config.delta_late+t)*H_now/c_s_now)
                            if ks[-1]>limit_late:
                                threshold_late = np.where(ks>limit_late)[0][0]
                            else:
                                threshold_late = Nk
                            if limit_early<limit_late:
                                limit_early=limit_late
                        else:
                            threshold_early = 0
                            threshold_late = 0
                            limit_late = ks[0]

                        indices_to_change = np.where((changed==False)*(ks<limit_late))[0]
                        #pos_re = slice(Nk)
                        #pos_im = slice(Nk,2*Nk)
                        #vel_re = slice(2*Nk,3*Nk)
                        #vel_im = slice(3*Nk,4*Nk)
                        for ind in indices_to_change:
                            zeta = (fields[pos_re][ind]+1j*fields[pos_im][ind])*np.exp(-1j*ks[ind]*tau_s_now)*c_s_now/math.sqrt(2*eps_now)
                            fields[pos_re][ind] = zeta.real
                            fields[pos_im][ind] = zeta.imag
                            dzeta = (fields[vel_re][ind]+1j*fields[vel_im][ind])*np.exp(-1j*ks[ind]*tau_s_now)*c_s_now/math.sqrt(2*eps_now)
                            dzeta += -zeta*(1j*ks[ind])*(math.exp(-t)/H_now)*c_s_now
                            dzeta += zeta*(-0.5*eta_now+eps_s_now)
                            fields[vel_re][ind] = dzeta.real
                            fields[vel_im][ind] = dzeta.imag
                            changed[ind] = True
                            changed_time[ind] = t

                        odeint_t1 = time()
                        #eval_at_times = [t,t_target]
                        if t_target>N_start_integ-beta_margin:
                            eval_at_times = np.linspace(t,t_target,10, endpoint=True)
                        else:
                            eval_at_times = np.linspace(t,t_target,2, endpoint=True)
                        ## # Careful here with chaining this, don't want t twice.
                        deriv_y = np.zeros_like(fields)
                        full_soln = odeint(mode_evolution_cython,fields,eval_at_times,args=(ks,threshold_early,threshold_late,As,Bs,Ns,deriv_y),atol=zeta_atols/del_t,rtol=zeta_rtols)
                        odeint_t2 = time()
                        odeint_time += odeint_t2-odeint_t1
                        fields = full_soln[-1]

                        tau_s_now   = interps_list[9](t_target)
                        eps_now     = interps_list[1](t_target)
                        eps_s_now   = interps_list[2](t_target)
                        eta_now     = interps_list[3](t_target)
                        phi_now     = interps_list[6](t_target)
                        c_s_now     = interps_list[7](t_target)
                        H_now       = interps_list[8](t_target)

                        rails_slice = slice(threshold_early,None)

                        z_re = -(H_now/(math.sqrt(2*c_s_now**3)))*np.ones(len(ks[rails_slice]))
                        z_im = ks[rails_slice]*c_s_now*math.exp(-t_target)/(math.sqrt(2*c_s_now**3))
                        dz_re = (-eps_now-1.5*eps_s_now)*z_re
                        dz_im = (-1-0.5*eps_s_now)*z_im

                        fields[pos_re][rails_slice]	=  z_re
                        fields[pos_im][rails_slice] =  z_im
                        fields[vel_re][rails_slice] =  dz_re
                        fields[vel_im][rails_slice]	=  dz_im
                        set_time[rails_slice] = t_target

                        if t_target > N_start_integ-beta_margin:
                            ## # Better for memory
                            zs_now_temp[:]  = np.array(full_soln[1:,pos_re]).T+1j*np.array(full_soln[1:,pos_im]).T
                            dzs_now_temp[:] = np.array(full_soln[1:,vel_re]).T+1j*np.array(full_soln[1:,vel_im]).T
                            zs_now[:] = np.zeros_like(zs_now_temp)*1j
                            dzs_now[:] = np.zeros_like(dzs_now_temp)*1j

                            time_points    = eval_at_times[1:]
                            eps_points     = interps_list[1](time_points)
                            eps_s_points   = interps_list[2](time_points)
                            eta_points     = interps_list[3](time_points)
                            phi_points     = interps_list[6](time_points)
                            c_s_points     = interps_list[7](time_points)
                            H_points       = interps_list[8](time_points)
                            tau_s_points   = interps_list[9](time_points)
                            m_points       = np.exp(-time_points)*c_s_points*ks[changed==False][:,None]/H_points
                            zs_now[changed==False] 	= zs_now_temp[changed==False]*c_s_points/np.sqrt(2*eps_points)
                            dzs_now[changed==False]     = (dzs_now_temp[changed==False]-zs_now_temp[changed==False]*(1j*m_points+0.5*eta_points-eps_s_points))*c_s_points/np.sqrt(2*eps_points)
                            zs_now[changed==True] 	= zs_now_temp[changed==True]*np.exp(1j*ks[changed==True][:,None]*tau_s_points)
                            dzs_now[changed==True] 	= dzs_now_temp[changed==True]*np.exp(1j*ks[changed==True][:,None]*tau_s_points)

                            ## # With the copy it works
                            zs[count:count+len(time_points)]    = np.copy(zs_now.T)
                            dzs[count:count+len(time_points)]   = np.copy(dzs_now.T)
                            count   += len(time_points)

                        t_temp = t + del_t
                        del_t = max(dt_pref*np.exp(t_temp),early_del_t)
                        del_t = min(del_t,late_del_t*10)
                        t_target = t_temp + del_t    ## New del_t

                    arr_t_zs = t_zs
                    arr_zs = zs
                    arr_dzs = dzs
                    gc.collect()
                    #print("# Odeint time:",odeint_time)
                    sys.stdout.flush()

                    return ks,arr_zs,arr_dzs

                #step 1: determine Ns(t_init) for given scenario
                interps_temp, bkgd_temp, As, Bs = get_bkgd_test(0, phi_0, t_tot = 70, phi_e = 50, atol = 10**(-12), rtol = 10**(-12))

                H_interp_temp = interps_temp[8]
                cs_interp_temp = interps_temp[7]
                k_cross = 0.05 #pivot scale
                H_cross = H_interp_temp(70-del_N_cross)
                cs_cross = cs_interp_temp(70-del_N_cross)
                t_init = np.log(k_cross*cs_cross/H_cross*np.e**(del_N_cross-70))
                #print(t_init)

                #step 2: evolve background from t_init, then evolve curvature perturbation
                interps_list,bkgd_results,As,Bs = get_bkgd_test(t_init, phi_0, t_tot = 70, phi_e = 50, atol = 10**(-12), rtol = 10**(-12))
                t_zs,t_list = get_t_zs()

                kbars,lg_ws = leggauss(config.Nk)
                try: 
                    ks,arr_zs,arr_dzs = get_zks(kbars,t_zs,t_list,interps_list,As,Bs, bkgd_results[0])

                    #step 3: get power spectrum (at tau = 0), then use curve-fit to find As and ns
                    power_spect = np.zeros(len(arr_zs[-1]))
                    zs_now = arr_zs[-1]
                    for i in range(len(zs_now)):
                        power_spect[i] = np.abs(zs_now[i])**2/(2*np.pi**2)

                    def log_P(log10k, log10As, ns):
                        return log10As+(ns-1)*(log10k-np.log10(0.05))

                    popt, pcov = curve_fit(log_P, np.log10(ks), np.log10(power_spect))
                    As = 10**popt[0]
                    ns = popt[1]
                    
                    #cs at horizon crossing of pivot mode
                    cs_interp = interps_list[7]
                    cs_pivot = cs_interp(t_init+70-del_N_cross)
                    
                    #decompose zeta, zeta' w.r.t 1d basis
                    basis_funcs = np.copy(working_basis)
                    try:
                        I_coeffs_final,J_coeffs_final = func_timeout(time_max, decomp_zs, args = (ks,lg_ws,t_zs,arr_zs,arr_dzs, interps_list, basis_funcs,l_max, late_t_switch))
                        a = 1
                    except:
                        I_coeffs_final = np.array([999])
                        a = 0
                    
                    #integrate and tidy up
                    if a == 1:
                        try:
                            coeffs, basis_funcs = func_timeout(time_max, construct_and_integrate, args = (t_zs, I_coeffs_final, J_coeffs_final, interps_list, basis_funcs, l_max, final_basis, N_start_integ, beta_activation, beta_margin))
                            b = 1
                        except:
                            b = 0
                    else:
                        b = 0

                    #evaluate fnl
                    if b == 1:
                        fnl, fnl_std = fnl_estimator(coeffs, pmax = 30)
                    else:
                        fnl = 999
                        fnl_std = 999
                    
                except:
                    As = 999
                    ns = 999
                    fnl = 999
                    fnl_std = 999
                    cs_pivot = 999
                
                # now we send our result back to the master
                comm.isend(np.array([As, ns, fnl, fnl_std, cs_pivot, task]), dest=0, tag=rank)
                #print("[{}] Sending result {} to rank {}".format(rank,result,0))

    return result_array


t1 = time()
scen_arr = np.loadtxt(scen_file)
result_array = bispectra_scan_ms(scen_arr)
t2 = time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0: 
    print(result_array)
    #np.save('omega_'+str(grids)+'_'+str(len(nijk_array))+'_'+str(len(mijk_array)), omega_array)
    np.savetxt('bispectrum_scan_result_fix_2', result_array)

print("time consuming "+str(t2-t1)) 