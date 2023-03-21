import numpy as np
import math
from scipy.integrate import tplquad
from scipy.integrate import quad
from scipy.special import eval_legendre
import general_coeff_funcs as gcf
from mpi4py import MPI
from time import time

#set everything needed here
kmin = 1.391727e-04 
kmax = 1.391727e-01
Nk_aug = 2500
pmax = 2 #for primodal basis
planck_file = "Basis_N200.txt" #n-ijk
line_remove = 208 #9 #from which line to read the file n-ijk
dk = kmax-kmin
K = kmax+kmin
log = False #generate primodal log basis, otherwise, generate primodal flat basis.

#generate 1d primodal basis
if log == True:
    pri_basis_1d = gcf.set_up_log_basis(kmin, kmax, pmax, inv=True, verbose=True)
else: 
    pri_basis_1d = gcf.set_up_flat_basis(kmin, kmax, pmax, Nk=Nk_aug, normalise=True)

#functions(internal use)
#calculate kbar from x, y, z (x = k/kmax)
def kbar(x):
    return 2*x*kmax/dk -K/dk

#generate 1d planck basis
def one_d_basis_f(i, x):
    m = math.fmod(i,2)
    n = i+m
    if m == 0:
        return math.cos(np.pi*n*x)
    else:
        return math.sin(np.pi*n*x)
    
#generate 3d planck basis (symmetrized over permutation)
def three_d_basis_f_eval(i,j,k, x1, x2, x3):
    
    res = one_d_basis_f(i, x1)*one_d_basis_f(i, x2)*one_d_basis_f(i, x3)
    res += one_d_basis_f(i, x2)*one_d_basis_f(i, x3)*one_d_basis_f(i, x1)
    res += one_d_basis_f(i, x3)*one_d_basis_f(i, x1)*one_d_basis_f(i, x2)
    res += one_d_basis_f(i, x3)*one_d_basis_f(i, x2)*one_d_basis_f(i, x1)
    res += one_d_basis_f(i, x2)*one_d_basis_f(i, x1)*one_d_basis_f(i, x3)
    res += one_d_basis_f(i, x1)*one_d_basis_f(i, x3)*one_d_basis_f(i, x2)

    return res/6

#input: filename. Read the file and return nijk_array, each line of which corresponds to a mapping from n to ijk
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

#map ijk to m, return the array(m,i,j,k)
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

#weight function, in order for excluding small x region in tetrahedral domain. (Non-observable scale)
def w(x,y,z,xmin = 0.001):
    if (x > xmin)*(y > xmin)*(z > xmin):
        weight = 1
    else:
        weight = 0
    return weight

#calculate the n-m element of Gamma matrix (integrate basises product over tetrahedral domain). n: planck, m: primodal
def Gamma_n_m(n_i, n_j, n_k, m_i, m_j, m_k, kmax): 
    #internal use, n is the index of 3d planck basis, m is the index of 3d primodal basis
    
    #doing the integration over tetrapyed (excluding the corner)
    f1 = lambda z, x, y: w(x,y,z)*three_d_basis_f_eval(n_i,n_j,n_k, x, y, z)*pri_basis_1d[m_i](kbar(x))*pri_basis_1d[m_j](kbar(y))*pri_basis_1d[m_k](kbar(z))
    f2 = lambda z, y, x: w(x,y,z)*three_d_basis_f_eval(n_i,n_j,n_k, x, y, z)*pri_basis_1d[m_i](kbar(x))*pri_basis_1d[m_j](kbar(y))*pri_basis_1d[m_k](kbar(z))
    
    I1, E1 = tplquad(f1, 0, 0.5, lambda y: y, lambda y: 1-y, lambda y, x: x-y, lambda x, y: x+y) #in the limit lambda, the sequence of
    I2, E2 = tplquad(f1, 0.5, 1, lambda y: 1-y, lambda y: y, lambda y, x: y-x, 1)                # y and x matters!
    I3, E3 = tplquad(f2, 0, 0.5, lambda x: x, lambda x: 1-x, lambda x, y: y-x, lambda x, y: x+y)
    I4, E4 = tplquad(f2, 0.5, 1, lambda x: 1-x, lambda x: x, lambda x, y: x-y, 1)

    return (I1+I2+I3+I4)*kmax**3  

#use for-loop to calculate Gamma n-m array
def Gamma_array(): #m: primodal, n: planck
    
    nijk_array = nijk_map(planck_file, line_remove)
    mijk_array = m_ijk(pmax) 
    Gamma_n_m_array = np.zeros((len(nijk_array), len(mijk_array))) #2d array 2000X219
    Gamma_n_ijk_array = np.zeros((len(nijk_array), pmax, pmax, pmax)) # 4d array 2000x10x10x10
    
    for n in range(len(nijk_array)):
        for m in range(len(mijk_array)):
            n_i = nijk_array[n][1]
            n_j = nijk_array[n][2]
            n_k = nijk_array[n][3]
            m_i = mijk_array[m][1]
            m_j = mijk_array[m][2]
            m_k = mijk_array[m][3]
            
            Gamma_n_m_array[n][m] = Gamma_n_m(n_i, n_j, n_k, m_i, m_j, m_k, kmax)

            Gamma_n_ijk_array[n][m_i][m_j][m_k] = Gamma_n_m_array[n][m] #using symmetry over ijk
            Gamma_n_ijk_array[n][m_k][m_i][m_j] = Gamma_n_m_array[n][m]
            Gamma_n_ijk_array[n][m_j][m_k][m_i] = Gamma_n_m_array[n][m]
            Gamma_n_ijk_array[n][m_i][m_k][m_j] = Gamma_n_m_array[n][m]
            Gamma_n_ijk_array[n][m_k][m_j][m_i] = Gamma_n_m_array[n][m]
            Gamma_n_ijk_array[n][m_j][m_i][m_k] = Gamma_n_m_array[n][m]

    return Gamma_n_m_array, Gamma_n_ijk_array

#Calculate Gamma n-m array with parallel method (divide the loop up...)
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
    
        Gamma_n_m_loop[n][m] = Gamma_n_m(n_i, n_j, n_k, m_i, m_j, m_k, kmax)

        Gamma_n_ijk_loop[n][m_i][m_j][m_k] = Gamma_n_m_loop[n][m] #using symmetry over ijk
        Gamma_n_ijk_loop[n][m_k][m_i][m_j] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_j][m_k][m_i] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_i][m_k][m_j] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_k][m_j][m_i] = Gamma_n_m_loop[n][m]
        Gamma_n_ijk_loop[n][m_j][m_i][m_k] = Gamma_n_m_loop[n][m]

    print( "[{}] Local sum {}".format(rank, Gamma_n_m_loop))
    Gamma_n_m_array = comm.reduce(Gamma_n_m_loop,op=MPI.SUM,root=0) #? does comm.reduce works for numpy array?
    Gamma_n_ijk_array = comm.reduce(Gamma_n_ijk_loop,op=MPI.SUM,root=0)

    return Gamma_n_m_array, Gamma_n_ijk_array

#main process
nijk_array = nijk_map(planck_file, line_remove)
mijk_array = m_ijk(pmax)

t1 = time()
Gamma_n_m_array, Gamma_n_ijk_array = Gamma_array_MIP()
#Gamma_n_m_array, Gamma_n_ijk_array = Gamma_array()
t2 = time()

print(Gamma_n_m_array)
print(Gamma_n_ijk_array)
print("time consuming "+str(t2-t1))
