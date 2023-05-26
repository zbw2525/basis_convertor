#define the integrating weight over the cube (0 if outside the tetrapyd), which depends on the volume of the grid cube inside and outside the tetrapyd.
#This code will generate a three-d matrix of weight values, only depends on the number of grid points, and is independent to the range of k (wavenumber).
import numpy as np
import numba as nb

@nb.njit
def cell1(a,b,c,d,e,f,g,h):
    result = (11*(a+b+c+e)+17*(d+g+f)+25*h)/240.0
    return result

@nb.njit
def cell2(a,b,c,d,e,f,g,h):
    result = (47*a+19*(b+c+e)+5*(d+f+g)+h)/720.0
    return result

@nb.njit
def cell3(a,b,c,d,e,f,g,h):
    result = (40*(b+c)+35*(a+d)+26*(f+g)+19*(e+h))/360.0
    return result

@nb.njit
def cell4(a,b,c,d,e,f,g,h):
    result = (89*a+85*(b+c+e)+71*(d+f+g)+43*h)/720.0
    return result

@nb.njit
def cell5(a,b,c,d,e,f,g,h):
    result = (a+b+c+d+e+f+g+h)/8.0
    return result
  

@nb.njit
def calculate_weight(min, max, i, j, k):
    grid = np.zeros((3,3,3))
    pt = np.zeros((3,3,3))
    weight = 0e0

    sum = 0 # sum over all neighbours (27 points)
    for l in range(3):
        i0 = i+l-1
        if i0 < min or i0 > max:
            for m in range(3):
                for n in range(3):
                    grid[l][m][n] = -8
        else:
            for m in range(3):
                j0 = j+m-1
                if j0 < min or j0 > max:
                    for n in range(3):
                        grid[l][m][n] = -8
                else:
                    for n in range(3):
                        k0 = k+n-1
                        if k0 < min or k0 > max:
                            grid[l][m][n] = -8
                        elif i0>(j0+k0) or j0>(i0+k0) or k0>(i0+j0):
                            grid[l][m][n] = 0
                        else:
                            grid[l][m][n] = 1
                            sum += 1
                        
    if sum == 27:
        weight = 1e0
    else:
        for l in range(3):
            for m in range(3):
                for n in range(3):
                 pt[l][m][n] = 0e0
        pt[1][1][1] = 1

        for s1 in range(2):
            for s2 in range(2):
                for s3 in range(2):
                    
                    sum = 0 # sum over a cube (8 points)
                    for l in range(2):
                        for m in range(2):
                            for n in range(2):
                                sum += grid[l+s1][m+s2][n+s3]
					
                    a = pt[s1][s2][s3]
                    b = pt[s1+1][s2][s3]
                    c = pt[s1][s2+1][s3]
                    d = pt[s1+1][s2+1][s3]
                    e = pt[s1][s2][s3+1]
                    f = pt[s1+1][s2][s3+1]
                    g = pt[s1][s2+1][s3+1]
                    h = pt[s1+1][s2+1][s3+1]

                    if sum == 4:
                        if grid[1+s1][0+s2][1+s3] == 1:
                            weight += cell2(f,e,h,g,b,a,d,c)
                        if grid[1+s1][1+s2][0+s3] == 1:
                            weight += cell2(d,c,b,a,h,g,f,e)
                        if grid[0+s1][1+s2][1+s3] == 1:
                            weight += cell2(g,h,e,f,c,d,a,b)
                    
                    elif sum == 5:
                        weight += cell1(a,b,c,d,e,f,g,h)

                    elif sum == 6:
                        if grid[0+s1][0+s2][1+s3] == 1:
                            weight += cell3(g,h,e,f,c,d,a,b)
                        if grid[1+s1][0+s2][0+s3] == 1:
                            weight += cell3(f,h,b,d,e,g,a,c)
                        if grid[0+s1][1+s2][0+s3] == 1:
                            weight += cell3(d,h,c,g,b,f,a,e)
                        
                    elif sum == 7:
                        if grid[0+s1][1+s2][0+s3] == 0:
                            weight += cell4(f,e,h,g,b,a,d,c)
                        if grid[0+s1][0+s2][1+s3] == 0:
                            weight += cell4(d,c,b,a,h,g,f,e)
                        if grid[1+s1][0+s2][0+s3] == 0:
                            weight += cell4(g,h,e,f,c,d,a,b)
                    
                    elif sum ==8:
                        weight += 1/8
    if i!=k:
        if i == j or j == k:
            weight *= 3e0
        else:
            weight *= 6e0
    
    return weight
  
#the integrating weight of each grid point in the cube.
weight_array = np.zeros((400, 400, 400))
for i in range(400):
    for j in range(400):
        for k in range(400):
            weight_array[i][j][k] = calculate_weight(0, 399, i, j, k)
