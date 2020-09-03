from pyfinite import ffield
import numpy as np
import copy
import random

def ns(Y,m,d = 0):
    "finds null space of a matrix A over the finite field F(2^m)"
    A = copy.copy(Y)
    F = ffield.FField(m)
    n,k = A.shape
    if type(d) == int:
        c = np.zeros(n,dtype = int)
    else:
        c = copy.copy(d)
    for j in range(min(n,k)):
        zeros = True
        for i in range(j,n): # make sure A[j][j] != 0 if it's possible
            if zeros == True and A[i][j] != 0: # swap
                B = copy.copy(A[i])
                x = copy.copy(c[i])
                C = copy.copy(A[j])
                y = copy.copy(c[j])
                A[i] = C
                c[i] = y
                A[j] = B
                c[j] = x
                zeros = False
        if A[j][j] != 0:
            B = copy.copy(A[j])
            for i in range(j,k):
                B[i] = F.Multiply(A[j][i],F.Inverse(A[j][j])) # normalizing row
            A[j] = B
            c[j] = F.Multiply(c[j],F.Inverse(A[j][j]))
            for i in range(n): # canceling corresponding entries in all other rows
                if i != j: 
                    B = copy.copy(A[i])
                    for l in range(k):
                        B[l] = F.Add(A[i][l],F.Multiply(A[j][l],A[i][j]))
                    A[i] = B
                    c[i] = F.Add(c[i],F.Multiply(c[j],A[i][j]))
    M = []
    indep = [set(range(k)) for z in range(n)]
    for i in range(n):
        j1 = 0
        while j1 < k-1 and A[i][j1] == 0:
            j1 += 1
        if j1 == k-1 and A[i][k-1] == 0 and c[i] != 0: # contradiction equation 0 != 0
            return "no solution!"
        if j1 < k-1 or (j1 == k-1 and A[i][j1] != 0): # nonzero row
            for j2 in indep[i]:
                vanishes = False
                if j2 > j1 and A[i][j2] != 0:
                    x = np.zeros(k,dtype = int)
                    x[j2] = 1
                    x[j1] = A[i][j2]
                    for i0 in range(i+1,n):
                        if A[i0][j2] != 0:
                            indep[i0].remove(j2)
                            j0 = 0
                            while A[i0][j0] == 0:
                                j0 += 1
                            if j0 == j2:
                                vanishes = True
                            x[j0] = A[i0][j2]
                    if not vanishes:
                        M.append(x)
    for j in range(k):
        if max(A.T[j]) == 0:
            x = np.zeros(k,dtype=int)
            x[j] = 1
            M.append(x)
    if type(d) != int:
        print(c)
    return np.asarray(M)

def inv(Y,m):
    "finds inverse of a square matrix A over the finite field F(2^m)"
    A = copy.copy(Y)
    if ns(A,m).shape[0] != 0: # nontrivial nullspace - not invertible!
        return "not invertible"
    F = ffield.FField(m)
    n = A.shape[0]
    X = np.zeros((n,n),dtype = int)
    for i in range(n):
        X[i][i] = 1
    for j in range(n):
        zeros = True
        for i in range(j,n):
            if zeros == True and A[i][j] != 0:
                B = copy.copy(A[i])
                B1 = copy.copy(X[i])
                C = copy.copy(A[j])
                C1 = copy.copy(X[j])
                A[i] = C
                A[j] = B
                X[i] = C1
                X[j] = B1
                zeros = False
        if A[j][j] != 0:
            B = copy.copy(A[j])
            B1 = copy.copy(X[j])
            for i in range(n):
                B[i] = F.Multiply(A[j][i],F.Inverse(A[j][j]))
                B1[i] = F.Multiply(X[j][i],F.Inverse(A[j][j]))
            A[j] = B
            X[j] = B1
            for i in range(n):
                if i != j:
                    B = copy.copy(A[i])
                    B1 = copy.copy(X[i])
                    for l in range(n):
                        B[l] = F.Add(A[i][l],F.Multiply(A[j][l],A[i][j]))
                        B1[l] = F.Add(X[i][l],F.Multiply(X[j][l],A[i][j]))
                    A[i] = B
                    X[i] = B1
    return X

def mul(A,B,m):
    "finite-field multiplication"
    F = ffield.FField(m)
    n1,m1 = A.shape
    n2,m2 = B.shape
    if m1 != n2:
        return "incompatible dimensions"
    C = np.zeros((n1,m2),dtype = int)
    for i in range(n1):
        for j in range(m2):
            cnt = 0
            for k in range(m1):
                cnt = F.Add(cnt,F.Multiply(A[i][k],B[k][j]))
            C[i][j] = cnt
    return C

def smul(x,A,m):
    "scalar matrix multiplication"
    F = ffield.FField(m)
    n,m = A.shape
    B = np.zeros((n,m),dtype = int)
    for i in range(n):
        for j in range(m):
            B[i][j] = F.Multiply(A[i][j],x)
    return B

def longmul(l,m):
    "recursive multiplication of a list of matrices"
    if len(l) == 1:
        return l[0]
    return mul(l[0],longmul(l[1:],m),m)

def invble(A,m):
    "checking whether a matrix is invertible"
    return type(inv(A,m)) != str

def sylmat(A,B,m,c,x1=0,y1=0):
    "finds a basis for the solution space to the sylvester equation AX = XB"
    "m is field size, c is subgroup, x1,y1 are sizes of upper free block in the subgroups"
    x2 = 0
    y2 = 0
    F = ffield.FField(m)
    n = A.shape[0]
    C = np.zeros((2*n**2,n**2), dtype = int)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i*n+j][k*n+j] = F.Add(C[i*n+j][k*n+j],A[i][k])
                C[i*n+j][i*n+k] = F.Add(C[i*n+j][i*n+k],B[k][j])
    l = list(range(n**2))
    if c == "A":
        for i in range(1,n-x1):
            C[n**2+i][x1*n+x1] = 1
            C[n**2+i][(x1+i)*n+x1+i] = 1
        for i in range(x1):
            for j in range(x1):
                l.remove(i*n+j)
        for i in range(x1,n):
            l.remove(i*n+i)
        for i in range(n,n):
            for j in range(n,n):
                l.remove(i*n+j)
    if c == "Y":
        for i in range(1,x1):
            C[n**2+i][0] = 1
            C[n**2+i][i*n+i] = 1
        for i in range(x1):
            l.remove(i*n+i)
        for i in range(x1,n):
            for j in range(x1,n):
                l.remove(i*n+j)
    if c == "B":
        for i in range(1,n-2*y1):
            C[n**2+i][y1*n+y1] = 1
            C[n**2+i][(y1+i)*n+y1+i] = 1
        for i in range(y1):
            for j in range(y1):
                l.remove(i*n+j)
            for j in range(n-y1,n):
                l.remove(i*n+j)
        for i in range(y1,n-y1):
            l.remove(i*n+i)
        for i in range(n-y1,n):
            for j in range(y1):
                l.remove(i*n+j)
            for j in range(n-y1,n):
                l.remove(i*n+j)
    if c == "X":
        for i in range(1,y1):
            C[n**2+i][0] = 1
            C[n**2+i][i*n+i] = 1
            C[n**2+y1+i][(n-y1)*n+n-y1] = 1
            C[n**2+y1+i][(n-y1+i)*n+n-y1+i] = 1
        for i in range(y1):
            l.remove(i*n+i)
        for i in range(y1,n-y1):
            for j in range(y1,n-y1):
                l.remove(i*n+j)
        for i in range(n-y1,n):
            l.remove(i*n+i)
    if c == "G":
        l = list()
    if c == "D":
        for i in range(n):
            l.remove(i*n+i)
    for i in range(len(l)):
        C[int(n**2+n+i)][l[i]] = 1
    null_vec = ns(C,m)
    a = null_vec.shape[0]
    null = []
    for i in range(len(null_vec)):
        null.append(null_vec[i].reshape((n,n)))
    return a,null

def syl(A,B,m,c,x1=0,y1=0):
    "tries to find an invertible solution to the sylvester equation AX = XB"
    "m is field size, c is subgroup"
    F = ffield.FField(m)
    n = A.shape[0]
    a,null = sylmat(A,B,m,c,x1,y1)
    if a == 0:
        return "no solution"
    f = lambda i:null[i].reshape((n,n))
    for cnt in range(20):
        ran = np.zeros((n,n),dtype = int)
        for j in range(a):
            c = np.random.randint(0,2**m)
            for i1 in range(n):
                for i2 in range(n):
                    ran[i1][i2] = F.Add(ran[i1][i2],F.Multiply(c,f(j)[i1][i2]))
        if invble(ran,m):
            return ran
    return "no invertible solution"

def findconj(A,m,c,x1=0,y1=0):
    "tries to find an invertible solution to the equation AX = YA"
    "m is field size, c is subgroup"
    x2 = 0
    y2 = 0
    F = ffield.FField(m)
    n = A.shape[0]
    C = np.zeros((3*n**2,2*n**2), dtype = int)
    # 3n^2 constraints: n^2 of equation and 2n^2 of subgroups. 2n^2 variables
    for i in range(n):
        for j in range(n):
        # equation i*n+j, which corresponds to the vanishing entry (i,j) in the product
            for k in range(n):
            # running over all variables that participate in this equation
                C[i*n+j][k*n+j] = F.Add(C[i*n+j][k*n+j],A[i][k])
                C[i*n+j][n**2 + i*n+k] = F.Add(C[i*n+j][n**2 + i*n+k],A[k][j])
    l = list(range(n**2))
    if c == "A":
        for i in range(1,n-x1):
            C[n**2+i][x1*n+x1] = 1
            C[n**2+i][(x1+i)*n+x1+i] = 1
            C[2*n**2+i][n**2+x1*n+x1] = 1
            C[2*n**2+i][n**2+(x1+i)*n+x1+i] = 1
        for i in range(x1):
            for j in range(x1):
                l.remove(i*n+j)
        for i in range(x1,n):
            l.remove(i*n+i)
    if c == "Y":
        for i in range(1,x1):
            C[n**2+i][0] = 1
            C[n**2+i][i*n+i] = 1
            C[2*n**2+i][n**2] = 1
            C[2*n**2+i][n**2+i*n+i] = 1
        for i in range(x1):
            l.remove(i*n+i)
        for i in range(x1,n):
            for j in range(x1,n):
                l.remove(i*n+j)
    if c == "B":
        for i in range(1,n-2*y1):
            C[n**2+i][y1*n+y1] = 1
            C[n**2+i][(y1+i)*n+y1+i] = 1
            C[2*n**2+i][n**2+y1*n+y1] = 1
            C[2*n**2+i][n**2+(y1+i)*n+y1+i] = 1
        for i in range(y1):
            for j in range(y1):
                l.remove(i*n+j)
            for j in range(n-y1,n):
                l.remove(i*n+j)
        for i in range(y1,n-y1):
            l.remove(i*n+i)
        for i in range(n-y1,n):
            for j in range(y1):
                l.remove(i*n+j)
            for j in range(n-y1,n):
                l.remove(i*n+j)
    if c == "X":
        for i in range(1,y1):
            C[n**2+i][0] = 1
            C[n**2+i][i*n+i] = 1
            C[2*n**2+i][n**2] = 1
            C[2*n**2+i][n**2+i*n+i] = 1
            C[n**2+y1+i][(n-y1)*n+n-y1] = 1
            C[n**2+y1+i][(n-y1+i)*n+n-y1+i] = 1
            C[2*n**2+y1+i][n**2+(n-y1)*n+n-y1] = 1
            C[2*n**2+y1+i][n**2+(n-y1+i)*n+n-y1+i] = 1            
        for i in range(y1):
            l.remove(i*n+i)
        for i in range(y1,n-y1):
            for j in range(y1,n-y1):
                l.remove(i*n+j)
        for i in range(n-y1,n):
            l.remove(i*n+i)
    if c == "G":
        l = list()
    if c == "D":
        for i in range(n):
            l.remove(i*n+i)
    for i in range(len(l)):
        C[n**2+n+i][l[i]] = 1
        C[2*n**2+n+i][n**2+l[i]] = 1
    null = ns(C,m)
    a,b = null.shape
    return a,null

def findcom(A,m,c,x1=0,y1=0):
    "finds all matrices that commute with A in subgroup c"
    return sylmat(A,A,m,c,x1,y1)

def asylmat(A,B,m,c,x1=0,y1=0):
    "finds a basis for the solution space to the equation AX = YB"
    "m is field size, c is subgroup"
    x2 = 0
    y2 = 0
    F = ffield.FField(m)
    n = A.shape[0]
    C = np.zeros((3*n**2,2*n**2), dtype = int)
    # 3n^2 constraints: n^2 of equation and 2n^2 of subgroups. 2n^2 variables
    for i in range(n):
        for j in range(n):
        # equation i*n+j, which corresponds to the vanishing entry (i,j) in the product
            for k in range(n):
            # running over all variables that participate in this equation
                C[i*n+j][k*n+j] = F.Add(C[i*n+j][k*n+j],A[i][k]) 
                C[i*n+j][n**2+i*n+k] = F.Add(C[i*n+j][n**2 + i*n+k],B[k][j])
    l = list(range(n**2))
    if c == "A":
        for i in range(1,n-x1):
            C[n**2+i][x1*n+x1] = 1
            C[n**2+i][(x1+i)*n+x1+i] = 1
            C[3*n**2+i][n**2+x1*n+x1] = 1
            C[3*n**2+i][n**2+(x1+i)*n+x1+i] = 1
        for i in range(x1):
            for j in range(x1):
                l.remove(i*n+j)
        for i in range(x1,n):
            l.remove(i*n+i)
    if c == "Y":
        for i in range(1,x1):
            C[n**2+i][0] = 1
            C[n**2+i][i*n+i] = 1
            C[3*n**2+i][n**2] = 1
            C[3*n**2+i][n**2+i*n+i] = 1
        for i in range(x1):
            l.remove(i*n+i)
        for i in range(x1,n):
            for j in range(x1,n):
                l.remove(i*n+j)
    if c == "B":
        for i in range(1,n-y1-y2):
            C[n**2+i][y1*n+y1] = 1
            C[n**2+i][(y1+i)*n+y1+i] = 1
            C[3*n**2+i][n**2+y1*n+y1] = 1
            C[3*n**2+i][n**2+(y1+i)*n+y1+i] = 1
        for i in range(y1):
            for j in range(y1):
                l.remove(i*n+j)
            for j in range(n-y2,n):
                l.remove(i*n+j)
        for i in range(y1,n-y2):
            l.remove(i*n+i)
        for i in range(n-y2,n):
            for j in range(y1):
                l.remove(i*n+j)
            for j in range(n-y2,n):
                l.remove(i*n+j)
    if c == "X":
        for i in range(1,y1):
            C[n**2+i][0] = 1
            C[n**2+i][i*n+i] = 1
            C[3*n**2+i][n**2] = 1
            C[3*n**2+i][n**2+i*n+i] = 1
        for i in range(1,y2):
            C[n**2+y1+i][(n-y2)*n+n-y2] = 1
            C[n**2+y1+i][(n-y2+i)*n+n-y2+i] = 1
            C[3*n**2+y1+i][n**2+(n-y2)*n+n-y2] = 1
            C[3*n**2+y1+i][n**2+(n-y2+i)*n+n-y2+i] = 1            
        for i in range(y1):
            l.remove(i*n+i)
        for i in range(y1,n-y2):
            for j in range(y1,n-y2):
                l.remove(i*n+j)
        for i in range(n-y2,n):
            l.remove(i*n+i)
    if c == "G":
        l = list()
    if c == "D":
        for i in range(n):
            l.remove(i*n+i)
    for i in range(len(l)):
        C[n**2+i][l[i]] = 1
        C[2*n**2+i][n**2+l[i]] = 1
    null = ns(C,m)
    a,b = null.shape
    return a,null

def asyl(A,B,m,c,x1=0,y1=0):
    "tries to find an invertible solution to the equation AX = YB"
    "m is field size, c is subgroup "    
    F = ffield.FField(m)
    n = A.shape[0]
    a,null = asylmat(A,B,m,c,x1,y1)
    if a == 0:
        return "no solution"
    f = lambda i:null[i][:n**2]
    g = lambda i:null[i][n**2:]
    for cnt in range(50):
        ran1 = np.zeros((n,n),dtype = int)
        ran2 = np.zeros((n,n),dtype = int)
        for j in range(a):
            c = np.random.randint(0,2**m)
            for i in range(n):
                for k in range(n):
                    ran1[i][k] = F.Add(ran1[i][k],F.Multiply(c,f(j)[i*n+k]))
                    ran2[i][k] = F.Add(ran2[i][k],F.Multiply(c,g(j)[i*n+k]))
        if invble(ran1,m) and invble(ran2,m):
            return ran1,ran2
    return "no invertible solution"
    
def ran(n,m,c,x1=0,y1=0):
    x2 = 0
    y2 = 0
    "generating random finite-field matrix"
    A = np.zeros((n,n),dtype = int)
    if c == "A":
        while not invble(A,m):
            for i in range(x1):
                for j in range(x1):
                    A[i][j] = np.random.randint(2**m)
            for i in range(n-x2,n):
                for j in range(n-x2,n):
                    A[i][j] = np.random.randint(2**m)    
            y = np.random.randint(1,2**m)
            for i in range(x1,n-x2):
                A[i][i] = y
    if c == "B":
        while not invble(A,m):
            for i in range(y1):
                for j in range(y1):
                    A[i][j] = np.random.randint(2**m)
                for j in range(n-y2,n):
                    A[i][j] = np.random.randint(2**m)
            for i in range(n-y2,n):
                for j in range(y1):
                    A[i][j] = np.random.randint(2**m)
                for j in range(n-y2,n):
                    A[i][j] = np.random.randint(2**m)    
            y = np.random.randint(1,2**m)
            for i in range(y1,n-y2):
                A[i][i] = y
    if c == "X":
        while not invble(A,m):
            for i in range(y1,n-y2):
                for j in range(y1,n-y2):
                    A[i][j] = np.random.randint(2**m)
            y = np.random.randint(1,2**m)
            for i in range(y1):
                A[i][i] = y
            y = np.random.randint(1,2**m)
            for i in range(n-y2,n):
                A[i][i] = y
    if c == "Y":
        while not invble(A,m):
            for i in range(x1,n-x2):
                for j in range(x1,n-x2):
                    A[i][j] = np.random.randint(2**m)
            y = np.random.randint(1,2**m)
            for i in range(x1):
                A[i][i] = y
            y = np.random.randint(1,2**m)
            for i in range(n-x2,n):
                A[i][i] = y
    if c == "G":
        while not invble(A,m):
            for i in range(n):
                for j in range(n):
                    A[i][j] = np.random.randint(2**m)
    if c == "D":
        for i in range(n):
            A[i][i] = np.random.randint(1,2**m)
    if c == "P":
        while not invble(A,m):
            alpha = np.random.randint(1,2**m)
            for i in range(0,n,2):
                for j in range(0,n,2):
                    A[i][j] = np.random.randint(0,2**m)
            for i in range(1,n,2):
                A[i][i] = alpha
    return A

def conj(A,B,m):
    "conjugating A by B"
    return longmul([inv(B,m),A,B],m)

def com(A,B,m):
    "commutator"
    return mul(inv(A,m),conj(A,B,m),m)

def experiment_4_1(n,m):
    M1 = ran(n,m,"G")
    M2 = ran(n,m,"G")
    a = ran(n,m,"D")
    b = ran(n,m,"D")
    x = ran(n,m,"D")
    y = ran(n,m,"D")
    a = conj(a,M1,m)
    b = conj(b,M2,m)
    x = conj(x,M2,m)
    y = conj(y,M1,m)
    K = com(a,b,m)
    u = conj(a,x,m)
    p = conj(b,y,m)
    ta = conj(inv(p,m),a,m)
    tb = conj(u,b,m)
    left = conj(inv(p,m),inv(M1,m),m)
    right = conj(ta,inv(M1,m),m)
    a_prime = syl(left,right,m,"D")
    left = conj(u,inv(M2,m),m)
    right = conj(tb,inv(M2,m),m)
    b_prime = syl(left,right,m,"D")
    a_prime = conj(a_prime,M1,m)
    b_prime = conj(b_prime,M2,m)
    K_prime = com(a_prime,b_prime,m)
    return (K == K_prime).all()

def lemma_3(n,m):
    M = ran(n,m,"G")
    a = ran(n,m,"D")
    a = conj(a,M,m)
    n = findcom(a,m,"D")[0]
    return n == 1

def experiment_4_2(n,m):
    M1 = ran(n,m,"G")
    M2 = ran(n,m,"G")
    M3 = ran(n,m,"G")
    M4 = ran(n,m,"G")
    a1 = conj(ran(n,m,"D"),M1,m)
    a2 = conj(ran(n,m,"D"),M2,m)
    b1 = conj(ran(n,m,"D"),M3,m)
    b2 = conj(ran(n,m,"D"),M4,m)
    x1 = conj(ran(n,m,"D"),M3,m)
    x2 = conj(ran(n,m,"D"),M4,m)
    y1 = conj(ran(n,m,"D"),M1,m)
    y2 = conj(ran(n,m,"D"),M2,m)
    K = longmul([a1,b1,a2,b2],m)
    u = longmul([inv(x1,m),a2,x2],m)
    p = longmul([inv(y1,m),b1,y2],m)
    ta = longmul([a1,p,a2],m)
    tb = longmul([b1,u,b2],m)
    left = longmul([M1,p,inv(M2,m)],m)
    right = longmul([M1,ta,inv(M2,m)],m)
    a2_prime,a1_inv_prime = asyl(left,right,m,"D")
    left = longmul([M3,u,inv(M4,m)],m)
    right = longmul([M3,tb,inv(M4,m)],m)
    b2_prime,b1_inv_prime = asyl(left,right,m,"D")
    a1_prime = conj(inv(a1_inv_prime,m),M1,m)
    a2_prime = conj(a2_prime,M2,m)
    b1_prime = conj(inv(b1_inv_prime,m),M3,m)
    b2_prime = conj(b2_prime,M4,m)
    K_prime = longmul([a1_prime,b1_prime,a2_prime,b2_prime],m)
    return (K==K_prime).all()

def lemma_7(n,m):
    M = ran(n,m,"G")
    a = ran(n,m,"D")
    a = conj(a,M,m)
    n = findconj(a,m,"D")[0]
    return n == 1

def experiment_4_3(n,m):
    F = ffield.FField(m)
    M1 = ran(n,m,"G")
    M2 = ran(n,m,"G")
    M3 = ran(n,m,"G")
    M4 = ran(n,m,"G")
    M5 = ran(n,m,"G")
    M6 = ran(n,m,"G")
    a0 = conj(ran(n,m,"D"),M1,m)
    a1 = conj(ran(n,m,"D"),M2,m)
    a2 = conj(ran(n,m,"D"),M3,m)
    b1 = conj(ran(n,m,"D"),M4,m)
    b2 = conj(ran(n,m,"D"),M5,m)
    b3 = conj(ran(n,m,"D"),M6,m)
    x0 = ran(n,m,"G")
    x1 = conj(ran(n,m,"D"),M4,m)
    x2 = conj(ran(n,m,"D"),M5,m)
    x3 = conj(ran(n,m,"D"),M6,m)
    y0 = conj(ran(n,m,"D"),M1,m)
    y1 = conj(ran(n,m,"D"),M2,m)
    y2 = conj(ran(n,m,"D"),M3,m)
    y3 = ran(n,m,"G")
    p = longmul([inv(y0,m),b1,y1],m)
    cnt = 0
    for n1 in range(1,2**m):
        for n2 in range(1,2**m):
            C = np.zeros((n**2,n**2), dtype = int)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        C[i*n+j][k] = F.Multiply(mul(M1,inv(M4,m),m)[i][k],mul(M4,inv(M2,m),m)[k][j])
            for j in range(n):
                C[0*n+j][n+j] = F.Multiply(n1,longmul([M1,p,inv(M2,m)],m)[i][j])
                C[1*n+j][n+j] = F.Multiply(n2,longmul([M1,p,inv(M2,m)],m)[i][j])
            for i in range(2,n):
                for j in range(n):
                    C[i*n+j][i*n+j] = longmul([M1,p,inv(M2,m)],m)[i][j]
            d = ns(C,m).shape[0]
            if d > 1:
                return False
            cnt += d
    return cnt

def experiment_5_3(n,m,n1,n2):
    k1 = max(n1,n2)
    k2 = min(n1,n2)
    M1 = ran(n,m,"G")
    M2 = ran(n,m,"G")
    a = ran(n,m,"A",x1=k1)
    b = ran(n,m,"A",x1=k2)
    x = ran(n,m,"Y",x1=k2)
    y = ran(n,m,"Y",x1=k1)
    a = conj(a,M1,m)
    b = conj(b,M2,m)
    x = conj(x,M2,m)
    y = conj(y,M1,m)
    K = com(a,b,m)
    u = conj(a,x,m)
    p = conj(b,y,m)
    ta = conj(inv(p,m),a,m)
    tb = conj(u,b,m)
    left = conj(u,inv(M2,m),m)
    right = conj(tb,inv(M2,m),m)
    b_prime = syl(left,right,m,"A",x1=k2)
    b_prime = conj(b_prime,M2,m)
    left = conj(b_prime,inv(M1,m),m)
    right = conj(p,inv(M1,m),m)
    c = 1
    while type(syl(smul(c,left,m),right,m,"Y",x1=k1)) == str:
        c += 1
    b_prime = smul(c,b_prime,m)
    left = smul(c,left,m)
    y_prime = syl(left,right,m,"Y",x1=k1)
    y_prime = conj(y_prime,M1,m)
    K_prime = longmul([y_prime,ta,inv(y_prime,m),b_prime],m)
    return (K == K_prime).all()

def lemma_3(n,m):
    M = ran(n,m,"G")
    a = ran(n,m,"D")
    a = conj(a,M,m)
    n = findcom(a,m,"D")[0]
    return n == 1
                    
    
    
    
    
    
    

