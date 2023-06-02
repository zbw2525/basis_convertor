#use grid integrator to evaluate the dot product between primodal basis and planck primodal basis

#!/usr/bin/env python3
import numpy as np
import numba as nb
from scipy.integrate import quad
from time import time
from mpi4py import MPI

#set everything needed
filename_cubic_weight = "cubic_weight_400.npy" #only need to change for different grid numbers
filename_weight_func = "weight_func_400.npy"
grids = 400

planck_file = "Basis_N2000.txt" #n-ijk
pmax_pl = 29 #maximum index ijk for planck primodal  basis
line_remove = 9 #from which line to read the file n-ijk

kmin = 1.391727e-04 
kmax = 1.391727e-01
pmax = 30 #for primodal basis, number of 1d modes
dk = kmax-kmin
K = kmax+kmin
log = True

#load cubic_weight_array
cubic_weight_array = np.load(filename_cubic_weight)

#load weight function array 1/(k1+k2+k3)
weight_func_evals = np.load(filename_weight_func)

#define and evaluate planck basis functions on the grid

#set planck mode functions
#generate 1d planck basis (fourier)
@nb.njit
def one_d_basis_f(i, x, pmax = pmax_pl):
    if x <= 0.001:
        result = 0
    else:
        if i == pmax: # check the expression for pmax, pmax-1 modes.If the argument is k, we need convert it to x.
            result = x**2
        elif i == pmax-1:
            result = 1/x
        else:
            m = np.mod(i,2)
            n = i+m
            if m == 0:
                result = np.cos(np.pi*n*x)
            else:
                result = np.sin(np.pi*n*x)
    return result


#evaluate planck basis over tetrapyd
@nb.njit
def eval_pl_basis_on_grid(points,i,j,k):
    eval_pl_array = np.zeros((points, points, points))
    x = np.linspace(0, 1, points)
    eval_i = np.zeros(points)
    eval_j = np.zeros(points)
    eval_k = np.zeros(points)
    for a in range(points):
       eval_i[a] = one_d_basis_f(i, x[a])
       eval_j[a] = one_d_basis_f(j, x[a])
       eval_k[a] = one_d_basis_f(k, x[a])

    for l in range(points):
        for m in range(l, points):
            for n in range(m, points):
                res = eval_i[l]*eval_j[m]*eval_k[n]
                res+= eval_i[m]*eval_j[n]*eval_k[l]
                res+= eval_i[n]*eval_j[l]*eval_k[m]
                res+= eval_i[l]*eval_j[n]*eval_k[m]
                res+= eval_i[n]*eval_j[m]*eval_k[l]
                res+= eval_i[m]*eval_j[l]*eval_k[n]
                eval_pl_array[l][m][n] = res/6

    return eval_pl_array

#primodal basis
#x to kbar
@nb.njit
def kbar(x):
    return 2*x*kmax/dk -K/dk

#kbar to k
@nb.njit
def unbar(kbar, kmin = kmin, kmax = kmax):
    return 0.5*((kmax-kmin)*kbar+kmin+kmax)
#legendre polynomial
@nb.njit
def legendre(n, x):
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        return ((2.0 * n - 1.0) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n 

#calculate the norm of pmax legendre basis 
def norm_cal_legendre(pmax):
    def legendre_n_sq(x,n):
        return (legendre(n, x))**2    
    
    norm = np.zeros(pmax)
    for i in range(pmax):
        n = i
        norm_sqr, error = quad(legendre_n_sq, -1, 1, args=(n,))
        norm[i] = np.sqrt(norm_sqr)
    return norm

#calculate the nomalization factors for legendre polynamial modes
legendre_norm = norm_cal_legendre(pmax)

#set primodal flat basis
@nb.njit
def pri_basis_legendre(n, x):
    return legendre(n,x)/legendre_norm[n]

#pri_basis_1: orth(1/k)
q_f1 = np.zeros(pmax-2)
def f1_dot_q(x,n):
    return 1/unbar(x)*pri_basis_legendre(n, x)
for i in range(pmax-2):
    n = i
    q_f1[i], error_i = quad(f1_dot_q, -1, 1, args=(n,))

@nb.njit
def orth_f1(x):
    res = 1/unbar(x)
    for i in range(pmax-2):
        res -= q_f1[i]*pri_basis_legendre(i, x)
    return res

@nb.njit
def f1_sq(x):
    return orth_f1(x)**2
norm_1_sq, error = quad(f1_sq, -1, 1)
norm_1 = np.sqrt(norm_1_sq)

@nb.njit
def pri_basis_1(x):
    return orth_f1(x)/norm_1

#pri_basis_0: orth(logk/k)
q_f0 = np.zeros(pmax-1)
def f0_dot_q(x,n):
    return np.log(unbar(x))/unbar(x)*pri_basis_legendre(n, x)

def f0_dot_f1(x):
    return np.log(unbar(x))/unbar(x)*pri_basis_1(x)

for i in range(pmax-1):
    n = i
    if i < pmax-2:
        q_f0[i], error_i = quad(f0_dot_q, -1, 1, args=(n,))
    else:
        q_f0[i], error_i = quad(f0_dot_f1, -1, 1)

@nb.njit
def orth_f0(x):
    res = np.log(unbar(x))/unbar(x)
    for i in range(pmax-1):
        if i < pmax-2:
            res -= q_f0[i]*pri_basis_legendre(i, x)
        else:
            res -= q_f0[i]*pri_basis_1(x)
    return res

@nb.njit
def f0_sq(x):
    return orth_f0(x)**2

norm_0_sq, error = quad(f0_sq, -1, 1)
norm_0 = np.sqrt(norm_0_sq)

@nb.njit
def pri_basis_0(x):
    return orth_f0(x)/norm_0

#this is the numba compatible primodal basis
@nb.njit
def pri_basis(n, x): 
    if x > 1e-10: #exclude the points with 0 coordinate in the grid
        x = kbar(x)
        if log == True:               
            if n == 0:
                value = pri_basis_0(x)
            elif n ==1:
                value = pri_basis_1(x)
            else:
                value = pri_basis_legendre(n-2, x)
        else:
            value = pri_basis_legendre(n, x)
    else:
        value = 0
    return value

#evaluate planck basis over tetrapyd
@nb.njit
def eval_pri_basis_on_grid(points,i,j,k):
    eval_pri_array = np.zeros((points, points, points))
    x = np.linspace(0, 1, points)
    eval_i = np.zeros(points)
    eval_j = np.zeros(points)
    eval_k = np.zeros(points)
    for a in range(points):
       eval_i[a] = pri_basis(i, x[a])
       eval_j[a] = pri_basis(j, x[a])
       eval_k[a] = pri_basis(k, x[a])

    for l in range(points):
        for m in range(l, points):
            for n in range(m, points):
                res = eval_i[l]*eval_j[m]*eval_k[n]
                res+= eval_i[m]*eval_j[n]*eval_k[l]
                res+= eval_i[n]*eval_j[l]*eval_k[m]
                res+= eval_i[l]*eval_j[n]*eval_k[m]
                res+= eval_i[n]*eval_j[m]*eval_k[l]
                res+= eval_i[m]*eval_j[l]*eval_k[n]
                eval_pri_array[l][m][n] = res/6

    return eval_pri_array

#map from n to ijk and ijk to m
#input: filename. Read the Planck file and return nijk_array, each line of which corresponds to a mapping from n to ijk
def nijk_map(filename, line_remove): # e.g."Basis_N2000.txt"
   
    with open(filename) as f:
        for _ in range(line_remove):
            next(f)
    
        lines = f.readlines()
        nijk_array = np.zeros((len(lines), 4))
        for i in range(len(lines)):
            data = list(map(int, lines[i].split('\t')))
            for j in range(4):
                nijk_array[i][j] = data[j]
    
    return nijk_array

#map ijk to m, return the array(m,i,j,k), for primodal basis
def m_ijk(pmax):
    mijk_list = []
    m = 0
    for i in range(pmax):
        for j in range(pmax):
            for k in range(pmax):
                if (i<=j)*(j<=k):
                    mijk_list.append([m,i,j,k])
                    m+=1
    return np.array(mijk_list)

#use master slave algorithm to avoid efficiency waste caused by un-uniform work load for each rank
def omega_array_ms(kmax, grids, nijk_array, mijk_array, cubic_weight_array, weight_func_evals):
    #initialize mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    #flatten NxM array to an 1-d array, and get the map between i to n-m
    omega_index_array = np.zeros((len(nijk_array), len(mijk_array),2))
    for n in range(len(nijk_array)):
        for m in range(len(mijk_array)):
            omega_index_array[n][m][0] = n
            omega_index_array[n][m][1] = m
    #1d map between the task index i and omega array index n,m.
    map_i_nm = omega_index_array.reshape(len(nijk_array)*len(mijk_array),2)

    #master rank (rank = 0): allocate tasks and integrate result.
    if rank==0:
        #set the empty array to restore the calculation result of omega matrix element
        omega_array = np.zeros((len(nijk_array), len(mijk_array)))
        
         # define list of tasks
        tasklist = (i for i in range(len(map_i_nm)))
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
                    n = int(map_i_nm[int(reply[1])][0])
                    m = int(map_i_nm[int(reply[1])][1])
                    omega_array[n][m] = reply[0]
                    try:
                        message = next(tasklist)
                    except StopIteration:
                        message = -1
                    print("[{}] Sending task: {} to rank {}".format(rank,message,source))
                    comm.isend(message, dest=source, tag=source)
        
    # Slave ranks section
    else:
        omega_array = np.zeros((len(nijk_array), len(mijk_array)))
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
                n = int(map_i_nm[task][0])
                m = int(map_i_nm[task][1]) 

                n_i = int(nijk_array[n][1])
                n_j = int(nijk_array[n][2])
                n_k = int(nijk_array[n][3])
                m_i = int(mijk_array[m][1])
                m_j = int(mijk_array[m][2])
                m_k = int(mijk_array[m][3])
                # These lines are the actual job
                pl_basis_evals = eval_pl_basis_on_grid(grids,n_i,n_j,n_k)
                pri_basis_evals = eval_pri_basis_on_grid(grids,m_i,m_j,m_k)
                max = grids-1
                result = np.sum(pl_basis_evals*pri_basis_evals*cubic_weight_array*weight_func_evals)*(kmax/max)**3
                
                # now we send our result back to the master
                comm.isend(np.array([result,task]), dest=0, tag=rank)
                #print("[{}] Sending result {} to rank {}".format(rank,result,0))

    return omega_array

#main
nijk_array = nijk_map(planck_file, line_remove)
mijk_array = m_ijk(pmax)
t1 = time()
omega_array =  omega_array_ms(kmax, grids, nijk_array, mijk_array, cubic_weight_array, weight_func_evals)
t2 = time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0: 
    print(omega_array)
    np.save('omega_'+str(grids)+'_'+str(len(nijk_array))+'_'+str(len(mijk_array)), omega_array)
    np.savetxt('omega_'+str(grids)+'_'+str(len(nijk_array))+'_'+str(len(mijk_array)), omega_array)

print("time consuming "+str(t2-t1)) 
