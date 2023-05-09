#speed up numerical integrations with numba.
#!/usr/bin/env python3
import numpy as np
from scipy.integrate import tplquad
from scipy.integrate import quad
import general_coeff_funcs as gcf
from mpi4py import MPI
from func_timeout import func_timeout, FunctionTimedOut
from time import time
import numba as nb

#set everything needed here
kmin = 1.391727e-04 
kmax = 1.391727e-01
pmax = 15 #for primodal basis, number of 1d modes
planck_file = "Basis_N500.txt" #n-ijk
pmax_pl = 29 #maximum index ijk for planck primodal  basis
line_remove = 9 #from which line to read the file n-ijk
dk = kmax-kmin
K = kmax+kmin
log = True
remove_bad = False #substitute the supplementary mode in planck basis with simple fourier mode.

atol = 1.49e-8
rtol = 1.49e-8
t_max = 360 #maximum time to calculate one element.
array_file_name = 'omega_n_m_501_680.txt' # name of the txt file of the restored omega array

#integral variable transiformation
#x to kbar
@nb.njit
def kbar(x):
    return 2*x*kmax/dk -K/dk

#kbar to k
@nb.njit
def unbar(kbar, kmin = kmin, kmax = kmax):
    return 0.5*((kmax-kmin)*kbar+kmin+kmax)


#set up planck mode functions
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
                result = np.cos(np.pi*n*x)
            else:
                result = np.sin(np.pi*n*x)
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


#set up primodal basis functions in a form compatible with numba.
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

#set primodal flat basis (normalized legendre polynomials)
@nb.njit
def pri_basis_legendre(n, x):
    return legendre(n,x)/legendre_norm[n]

#1st mode (augmented): orth(1/k), see arXiv:2012.08546 eqn 3.9
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

#0th mode (augmented): orth(logk/k)
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

#map from n to ijk for Planck basis and ijk to m for primodal basis
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

#weight function, used to determine the dot product between basis functions over tetrapyd. w = 1/(k1+k2+k3)
@nb.njit
def w(x,y,z, kmax = kmax):
    if (x > 0.001)*(y > 0.001)*(z > 0.001):
        w = 1/kmax*1/(x+y+z)
    else:
        w = 0
    return w

#numerical integrator
def Gamma_n_m_nb(n_i, n_j, n_k, m_i, m_j, m_k, kmax, atol = 1.49e-8, rtol = 1.49e-8): #internal use, n is the index of 3d planck basis, m is the index of 3d primodal basis
    
    #doing the integration over tetrapyed (excluding the small k slices)
    @nb.njit
    def f1(z,x,y):
        return w(x,y,z)*three_d_basis_f_eval(n_i,n_j,n_k, x, y, z)*pri_basis(m_i, kbar(x))*pri_basis(m_j, kbar(y))*pri_basis(m_k, kbar(z))
    @nb.njit    
    def f2(z,y,x):
        return w(x,y,z)*three_d_basis_f_eval(n_i,n_j,n_k, x, y, z)*pri_basis(m_i, kbar(x))*pri_basis(m_j, kbar(y))*pri_basis(m_k, kbar(z))
    
    I1, E1 = tplquad(f1, 0, 0.5, lambda y: y, lambda y: 1-y, lambda y, x: x-y, lambda x, y: x+y, epsabs=atol, epsrel=rtol) #in the limit lambda, the sequence of
    I2, E2 = tplquad(f1, 0.5, 1, lambda y: 1-y, lambda y: y, lambda y, x: y-x, 1, epsabs=atol, epsrel=rtol)                # y and x matters!
    I3, E3 = tplquad(f2, 0, 0.5, lambda x: x, lambda x: 1-x, lambda x, y: y-x, lambda x, y: x+y, epsabs=atol, epsrel=rtol)
    I4, E4 = tplquad(f2, 0.5, 1, lambda x: 1-x, lambda x: x, lambda x, y: x-y, 1, epsabs=atol, epsrel=rtol)

    return (I1+I2+I3+I4)*kmax**3

#calculate gamma (or omega matrix in new convention), with MIP parallelization
def Gamma_array_MIP(): #m: primodal, n: planck

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
    
    def gen(s,e,nmax,mmax):
        count = 1
        for n in range(nmax):
            for m in range(mmax):
                if count>e:
                    break
                if count>s:
                    yield (n,m)
                count += 1

    nijk_array = nijk_map(planck_file, line_remove)
    if remove_bad == True:
    #substitute the "bad" mode in Planck basis with a "Good" modes
        nijk_array[1] = [1,0,0,1]
    
    mijk_array = m_ijk(pmax)
    nmax = len(nijk_array) 
    mmax = len(mijk_array)
    
    # initilise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print( "[{}] Starting".format(rank) )

    #loops
    loops = nmax*mmax
    #divide up the loops
    s,e = divide_loops(loops,size)
    #create generator for these loops
    G1 = gen(s,e,nmax, mmax)

    #compute the gamma n-m element in this group
    Gamma_n_m_loop = np.zeros((nmax, mmax)) #2d array 2000X219, element with index within this divided loop are non-zero
    Gamma_n_ijk_loop = np.zeros((nmax, pmax, pmax, pmax)) # 4d array 2000x10x10x10
    for item in G1:
        n = item[0]
        m = item[1]

        n_i = int(nijk_array[n][1])
        n_j = int(nijk_array[n][2])
        n_k = int(nijk_array[n][3])
        m_i = int(mijk_array[m][1])
        m_j = int(mijk_array[m][2])
        m_k = int(mijk_array[m][3])
    
        #Gamma_n_m_loop[n][m] = Gamma_n_m(n_i, n_j, n_k, m_i, m_j, m_k, kmax)
        #if the calculation time exceed 1h, force it to terminate and return 999
        try:
            Gamma_n_m_loop[n][m] = func_timeout(360, Gamma_n_m_nb, args=(n_i, n_j, n_k, m_i, m_j, m_k, kmax))

        except FunctionTimedOut:
            Gamma_n_m_loop[n][m] = 999
            
        Gamma_n_ijk_loop[n][m_i][m_j][m_k] = Gamma_n_m_loop[n][m] #using symmetry over ijk
        Gamma_n_ijk_loop[n][m_k][m_i][m_j] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_j][m_k][m_i] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_i][m_k][m_j] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_k][m_j][m_i] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_j][m_i][m_k] = Gamma_n_m_loop[n][m]

    #print( "[{}] Local sum {}".format(rank, Gamma_n_m_loop))
    Gamma_n_m_array = comm.reduce(Gamma_n_m_loop,op=MPI.SUM,root=0) #? does comm.reduce works for numpy array?
    Gamma_n_ijk_array = comm.reduce(Gamma_n_ijk_loop,op=MPI.SUM,root=0)

    return Gamma_n_m_array, Gamma_n_ijk_array

#use master slave algorithm to avoid wasting calculating efficiency caused by non-uniform distributions of work load for each rank
def omega_array_ms(atol, rtol, time_max):
    #initialize mpi
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #get the map between n, m to ijk
    nijk_array = nijk_map(planck_file, line_remove)
    mijk_array = m_ijk(pmax)
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
            print("[{}] Recieving task: {} from rank {}".format(rank,task,0))
        
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
                try:
                    result = func_timeout(time_max, Gamma_n_m_nb, args=(n_i, n_j, n_k, m_i, m_j, m_k, kmax, atol, rtol))

                except FunctionTimedOut:
                    result = 999
                
                # now we send our result back to the master
                comm.isend(np.array([result,task]), dest=0, tag=rank)
                #print("[{}] Sending result {} to rank {}".format(rank,result,0))

    return omega_array

#main
t1 = time()
#omega_n_m_array, omega_n_ijk_array = Gamma_array_MIP()
omega_n_m_array = omega_array_ms(atol, rtol, t_max)
t2 = time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0: 
    print(omega_n_m_array)
    #print(omega_n_ijk_array)
    np.savetxt(array_file_name, omega_n_m_array)

print("time consuming "+str(t2-t1))
