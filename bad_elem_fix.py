#This code take the output array from "primo_dot_Planck_nb.py", identifying the "bad" elements, which failed to be 
#calculated in the time limit with desired tolerance and re-computing them with larger time limit and looser tolerance.

#!/usr/bin/env python3
import numpy as np
import math
from scipy.integrate import tplquad
from scipy.integrate import quad
import general_coeff_funcs as gcf
from mpi4py import MPI
from time import time
from func_timeout import func_timeout, FunctionTimedOut
import numba as nb 

#set everything needed here
atol = 1e-7 #Looser tolerance
rtol = 1e-7
time_max = 1800 #max time allowed for integration
matrix_temp_file = 'gamma_n_m_500_680.txt' #the file of gamma matrix with bad elements

#following setup must agree with that in "primo_dot_Planck_nb.py"
kmin = 1.391727e-04 
kmax = 1.391727e-01
pmax = 15 #for primodal basis, number of 1d modes
planck_file = "Basis_N500.txt" #n-ijk
pmax_pl = 29 #maximum index ijk for planck primodal  basis
line_remove = 9 #from which line to read the file n-ijk
dk = kmax-kmin
K = kmax+kmin
log = True
remove_bad = True #if true, we need to change the subsituted fourier mode back to supplementary mode

#argument transiformation
#x to kbar
@nb.njit
def kbar(x):
    return 2*x*kmax/dk -K/dk
#x to xbar = (k-kmin)/(kmax-kmin), which is the argument of the fourier modes in planck primodal basis with 1d domain (0,1)
@nb.njit
def xbar(x):
    return x*kmax/dk-kmin/dk
#kbar to k
@nb.njit
def unbar(kbar, kmin = kmin, kmax = kmax):
    return 0.5*((kmax-kmin)*kbar+kmin+kmax)

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
            result =  1/x
        else:
            m = np.mod(i,2)
            n = i+m
            if m == 0:
                result = np.cos(np.pi*n*xbar(x))
            else:
                result = np.sin(np.pi*n*xbar(x))
    return result

#generate 3d planck basis, then symmetrize.
@nb.njit
def three_d_basis_f_eval(i,j,k, x1, x2, x3):
    
    res = one_d_basis_f(i, x1)*one_d_basis_f(j, x2)*one_d_basis_f(k, x3)
    res += one_d_basis_f(i, x2)*one_d_basis_f(j, x3)*one_d_basis_f(k, x1)
    res += one_d_basis_f(i, x3)*one_d_basis_f(j, x1)*one_d_basis_f(k, x2)
    res += one_d_basis_f(i, x3)*one_d_basis_f(j, x2)*one_d_basis_f(k, x1)
    res += one_d_basis_f(i, x2)*one_d_basis_f(j, x1)*one_d_basis_f(k, x3)
    res += one_d_basis_f(i, x1)*one_d_basis_f(j, x3)*one_d_basis_f(k, x2)

    return res/6

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
    if log == True:               
        if n == 0:
            value = pri_basis_0(x)
        elif n ==1:
            value = pri_basis_1(x)
        else:
            value = pri_basis_legendre(n-2, x)
    else:
        value = pri_basis_legendre(n, x)
    return value

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

#numerical integrator
def Gamma_n_m_nb(n_i, n_j, n_k, m_i, m_j, m_k, kmax, atol = 1.49e-8, rtol = 1.49e-8): #internal use, n is the index of 3d planck basis, m is the index of 3d primodal basis
    
    #doing the integration over tetrapyed (excluding the small k slices)
    @nb.njit
    def f1(z,x,y):
        return three_d_basis_f_eval(n_i,n_j,n_k, x, y, z)*pri_basis(m_i, kbar(x))*pri_basis(m_j, kbar(y))*pri_basis(m_k, kbar(z))
    @nb.njit    
    def f2(z,y,x):
        return three_d_basis_f_eval(n_i,n_j,n_k, x, y, z)*pri_basis(m_i, kbar(x))*pri_basis(m_j, kbar(y))*pri_basis(m_k, kbar(z))
    
    I1, E1 = tplquad(f1, 0, 0.5, lambda y: y, lambda y: 1-y, lambda y, x: x-y, lambda x, y: x+y, epsabs=atol, epsrel=rtol) #in the limit lambda, the sequence of
    I2, E2 = tplquad(f1, 0.5, 1, lambda y: 1-y, lambda y: y, lambda y, x: y-x, 1, epsabs=atol, epsrel=rtol)                # y and x matters!
    I3, E3 = tplquad(f2, 0, 0.5, lambda x: x, lambda x: 1-x, lambda x, y: y-x, lambda x, y: x+y, epsabs=atol, epsrel=rtol)
    I4, E4 = tplquad(f2, 0.5, 1, lambda x: 1-x, lambda x: x, lambda x, y: x-y, 1, epsabs=atol, epsrel=rtol)

    return (I1+I2+I3+I4)*kmax**3

#read the matrix file, find bad elements(999) and output their indices as a numpy array
def bad_elem_finder(filename, remove_bad = remove_bad): 
    data = np.loadtxt(filename)
    if remove_bad == True:
        for i in range(len(data[1])):
            data[1][i] = 999
    bad_elem = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] == 999:
                bad_elem.append([i,j])
    bad_array = np.array(bad_elem)
    return bad_array

def bad_elem_calc(bad_array, atol, rtol, timelimit):
    def divide_loops(loops,size):
    
        # floor division
        loop_rank=loops//size
        # remainder
        auxloop = loops%size
        #calculate start and end
        start_loop = rank*loop_rank
        end_loop = (rank+1)*loop_rank
    
        # allocate remainder across loops
        if(auxloop!=0):
            if (rank < auxloop):
                start_loop = start_loop + rank
                end_loop = end_loop + rank + 1
            else:
                start_loop = start_loop + auxloop
                end_loop = end_loop + auxloop
        # return start and end
        return start_loop, end_loop

    nijk_array = nijk_map(planck_file, line_remove)
    mijk_array = m_ijk(pmax)
    
    # initilise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print( "[{}] Starting".format(rank) )

    #loops
    loops = len(bad_array)
    #divide up the loops
    s,e = divide_loops(loops,size)

    #compute the gamma n-m element in this group
    fixed_elem_loop = np.zeros(len(bad_array))#the 1d array of recalculated bad element, with loosen tolerence
    for i in range(s,e):
        n = bad_array[i][0]
        m = bad_array[i][1]

        n_i = int(nijk_array[n][1])
        n_j = int(nijk_array[n][2])
        n_k = int(nijk_array[n][3])
        m_i = int(mijk_array[m][1])
        m_j = int(mijk_array[m][2])
        m_k = int(mijk_array[m][3])

        try:
            fixed_elem_loop[i] = func_timeout(time_max, Gamma_n_m_nb, args=(n_i, n_j, n_k, m_i, m_j, m_k, kmax, atol, rtol))

        except FunctionTimedOut:
            fixed_elem_loop[i] = 999
    
    #print( "[{}] Local sum {}".format(rank, fixed_elem_loop))
    fixed_elem_array = comm.reduce(fixed_elem_loop,op=MPI.SUM,root=0) #? does comm.reduce works for numpy array?

    return fixed_elem_array

#calculate bad_elem, with master and slave parallelization process
def bad_elem_calc_ms(bad_array, atol, rtol, time_max):
    #initialize mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #get the map between n, m to ijk
    nijk_array = nijk_map(planck_file, line_remove)
    mijk_array = m_ijk(pmax)
    #the map between bad_array element index and n,m is given by bad_array

    #master rank (rank = 0): allocate tasks and integrate result.
    if rank==0:
        fixed_elem_array = np.zeros(len(bad_array))
        # define list of tasks
        tasklist = (i for i in range(len(bad_array)))
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
                    fixed_elem_array += reply
                    try:
                        message = next(tasklist)
                    except StopIteration:
                        message = -1
                    print("[{}] Sending task: {} to rank {}".format(rank,message,i))
                    comm.isend(message, dest=source, tag=source)
        
    # Slave ranks section
    else:
        fixed_elem_array = np.zeros(len(bad_array))
        # this loop keeps us checking for jobs until we recieve a -1
        while True:
            # recvout next job ID
            test = comm.irecv(source=0, tag=rank)
            task = test.wait()#task: job ID
            print("[{}] Recieving task: {} from rank {}".format(rank,task,0))
        
            # is job ID = -1 then no more jobs and we can stop
            # We let the master know we are stopping by sending -1 back
            # Otherwise we do the job associated with the ID we recieve
            if task==-1:
                comm.isend(np.array([-1]), dest=0, tag=rank)
                #print("[{}] Sending termination to rank {}".format(rank,0))
                break
            else:
                n = bad_array[task][0]
                m = bad_array[task][1]

                n_i = int(nijk_array[n][1])
                n_j = int(nijk_array[n][2])
                n_k = int(nijk_array[n][3])
                m_i = int(mijk_array[m][1])
                m_j = int(mijk_array[m][2])
                m_k = int(mijk_array[m][3])
                # This single line is the actual job
                result = np.zeros(len(bad_array))
                try:
                    result[task] = func_timeout(time_max, Gamma_n_m_nb, args=(n_i, n_j, n_k, m_i, m_j, m_k, kmax, atol, rtol))

                except FunctionTimedOut:
                    result[task] = 999
                
                # now we send our result back to the master
                comm.isend(result, dest=0, tag=rank)
                #print("[{}] Sending result {} to rank {}".format(rank,result,0))

    return fixed_elem_array

#main
bad_array = bad_elem_finder(matrix_temp_file)
#bad_array = np.array([[0,2],[0,3],[0,4],[0,5],[0,6]])
t1 = time()
fixed_elem_arr = bad_elem_calc_ms(bad_array, atol, rtol, time_max)
#fixed_elem_arr = bad_elem_calc(bad_array, atol, rtol, time_max)
t2 = time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0: 
    print(fixed_elem_arr)
    np.savetxt('elem_fixed_500_680.txt', fixed_elem_arr)

print("time consuming "+str(t2-t1)) 
