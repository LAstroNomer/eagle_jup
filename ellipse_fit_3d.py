import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from numpy.random import normal
from numpy.linalg import eig
from galactic_ellipse import fit2D, first_momentum, second_momentum, cross_momentum
import matplotlib as mpl
from radial_profiles_calc import const_numb_bins, const_width_bins

def momentum(x, y, z, velocity :np.ndarray, mass :np.ndarray, 
            rmin :float =0.0025, rmax :float =  0.03):
    I = np.zeros(3)
    

    d = np.array(np.sqrt(x**2 + y**2 + z**2))
    #d = np.array( [(xi/A)**2 + (yi/B)**2 + (zi/C)**2 for xi, yi, zi in zip(x, y,z)])
    
    #nd = np.where((d >= rmin ) * (d <= rmax))
    
    coord1    = [np.array([a, b, c ]) for a, b, c in zip(x, y ,z)]
    #velocity1 = velocity[ind]
    #mass1     = mass[ind]

    #plt.figure()
    #plt.scatter(x, y)
    #plt.show()
    for r, v, m in zip(coord1, velocity, mass):
        I += m * np.cross(r, v)
        
    return I 




def wcow(X, Y, m):
    xc = np.sum(X*m)/np.sum(m)
    yc = np.sum(Y*m)/np.sum(m)
    return np.sum((X-xc)*(Y-yc)*m)/np.sum(m)

def rot(a, b, g):
    ca = np.cos(np.radians(a))
    sa = np.sin(np.radians(a))

    cb = np.cos(np.radians(b))
    sb = np.sin(np.radians(b))

    cg = np.cos(np.radians(g))
    sg = np.sin(np.radians(g))

    R = np.array(
        [[ca*cg-sa*cb*sg, -ca*sg-sa*cb*cg, sa*sb],
        [sa*cg+ca*cb*sg, -sa*sg + ca*cb*cg,-ca*sb],
        [sb*sg, sb*cg, cb]]
    )
    return R

def prep(coord, Q, A, B, C):
    
    coord1 = np.array([Q.T@a for a in coord])
    X = coord1[:, 0]
    Y = coord1[:, 1]
    Z = coord1[:, 2]

    ind = np.where(X**2/A**2 + Y**2/B**2 + Z**2/C**2 <= 1)
    return ind

def solve(P):
    lamb, vecs = eig(P)

    vx = vecs[:, 0]
    vy = vecs[:, 1]
    vz = vecs[:, 2]
    Jxx, Jyy, Jzz = lamb
    
    J = np.array([[Jxx, 0, 0],[0, Jyy, 0],[0, 0, Jzz]])

    if (Jzz < Jyy) and (Jzz < Jxx):
        Q = vecs    
        vzz = vz    
    elif (Jxx < Jyy) and (Jxx < Jzz):
        R = np.array([[0,0,1], [0, 1, 0], [-1, 0, 0]])
        Q = vecs @ R
        J = R.T @ J @ R
        vzz = vx
        #print(R.transpose() @ T @ R)
    else:
        R = np.array([[1,0,0], [0, 0, 1], [0, -1, 0]])
        Q =  vecs @ R
        J = R.T @ J @ R
        vzz = vy
        
    #print(J)
    #print(vecs)
    #exit()
    
    Jxx = J[0,0]
    Jyy = J[1,1]
    #print(J)
    if Jxx < Jyy:
        Jxx, Jyy = Jyy, Jxx
        R = np.array([[0, -1, 0], [1, 0, 0], [0,0,1]])
        J = R.T @ J @ R
        Q = Q @ R

    return Q

def run(coord, mass, Q0, A, B, C):

    if not(A is None):
        ind = prep(coord, Q0, A, B, C)
        coord1 = coord[ind]
        mass   = mass[ind] 
    else:
        coord1 = coord

    X = coord1[:, 0]
    Y = coord1[:, 1]
    Z = coord1[:, 2]

    rp = np.array([X**2 + Y**2 ])

    X2 = wcow(X, X, mass)
    Y2 = wcow(Y, Y, mass)
    Z2 = wcow(Z, Z, mass)
    XY = wcow(X, Y, mass)
    XZ = wcow(X, Z, mass)
    YZ = wcow(Y, Z, mass)

    P = np.array([[X2, XY, XZ], [XY, Y2, YZ], [XZ, YZ, Z2]])
    lamb, vecs = eig(P)

    Jxx, Jyy, Jzz = lamb
    J = np.array([[Jxx, 0, 0],[0, Jyy, 0],[0, 0, Jzz]])

    #print('BEFORE')
    #print(J)

    if (Jzz < Jyy) and (Jzz < Jxx):
        Q = vecs
        
    elif (Jxx < Jyy) and (Jxx < Jzz):
        R = np.array([[0,0,1], [0, 1, 0], [-1, 0, 0]])
        Q = vecs @ R
        J = R.T @ J @ R
        #print(R.transpose() @ T @ R)
    else:
        R = np.array([[1,0,0], [0, 0, 1], [0, -1, 0]])
        Q =  vecs @ R
        J = R.T @ J @ R

    #print('Z rotate')
    Jxx = J[0,0]
    Jyy = J[1,1]
    #print(J)
    if Jxx < Jyy:
        Jxx, Jyy = Jyy, Jxx
        R = np.array([[0, -1, 0], [1, 0, 0], [0,0,1]])
        J = R.T @ J @ R
        Q = Q @ R

    #tx = np.arctan2(Q[2,1], Q[2,2])
    #ty = np.arctan2(-Q[2,0], np.sqrt(Q[2,2]**2 + Q[2,1]**2))
    #tz = np.arctan2(Q[1,0], Q[0,0])
    
    #print('XY rotate')
    #print(J)

    Jxx = J[0,0]
    Jyy = J[1,1]
    Jzz = J[2,2]

    s = np.sqrt(7.815) #95%
    #s = np.sqrt(11.34) # 99%
    A1 = np.sqrt(Jxx) * s
    B1 = np.sqrt(Jyy) * s
    C1 = np.sqrt(Jzz) * s

    #t = np.arange(0, 2*np.pi, 0.001)
    return Q, A1, B1, C1

def Gauss3D(n, A, B, C, a, b, g):


 
    X = normal(loc=0.0, scale=A, size = n)
    Y   = normal(loc=0.0, scale=B, size = n)
    Z = normal(loc=0.0, scale=C, size = n)

    r2 = X**2 + Y**2 + Z**2

    mass = np.ones(len(X)) #1*np.exp(-r2/1000)
    R = rot(a, b, g)
    coord = np.array([R@a for a in zip(X, Y, Z)])

    #print(fit2D(coord[:,0], coord[:,1], mass))

    #plt.figure()
    #plt.scatter(coord[:,0], coord[:,1], s=1)
    #plt.gca().set_aspect('equal')
    #plt.show()

    return coord, mass

def matr_notm(A):
    a_row = np.sum(abs(A), axis=0)
    return np.max(a_row)

def anlge_from_matrix(Q):

    E2 = - np.arcsin(Q[0,2])
    E1 = np.arctan2(Q[1,2]/np.cos(E2), Q[2,2]/np.cos(E2))
    E3 = np.arctan2(Q[0,1]/np.cos(E2), Q[0,0]/np.cos(E2))

    #tx = np.arctan2(Q[2,1], Q[2,2])
    #ty = np.arctan2(-Q[2,0], np.sqrt(Q[2,2]**2 + Q[2,1]**2))
    #tz = np.arctan2(Q[1,0], Q[0,0])
    return np.degrees(E1), np.degrees(E2), np.degrees(E3)

def itter_tensor(X, Y, Z, mass, app=0.03, maxiter=10):

    app_min = 0.0025

    X2 = wcow(X, X, mass)
    Y2 = wcow(Y, Y, mass)
    Z2 = wcow(Z, Z, mass)
    XY = wcow(X, Y, mass)
    XZ = wcow(X, Z, mass)
    YZ = wcow(Y, Z, mass)

    P = np.array([[X2, XY, XZ], [XY, Y2, YZ], [XZ, YZ, Z2]])
    #Jxx, Jyy, Jzz, Q, vz = solve(P)
    lamb, vecs = eig(P)
    Jxx, Jyy, Jzz = lamb
    a = np.sqrt(Jxx)
    b = np.sqrt(Jyy)
    c = np.sqrt(Jzz)

    vx = vecs[:,0]
    vy = vecs[:,1]
    vz = vecs[:,2]

    axes = sorted([(a, vx), (b, vy), (c, vz)])

    a, ax_a = axes[2]
    b, ax_b = axes[1]
    c, ax_c = axes[0]
   
    s1 = c/a
    q1 = b/a
    #print(np.sum(ax_a**2))

    rp1 = np.array([np.dot(ax_a, r) for r in zip(X, Y,Z)])
    rp2 = np.array([np.dot(ax_b, r) for r in zip(X, Y,Z)])
    rp3 = np.array([np.dot(ax_c, r) for r in zip(X, Y,Z)])

    rp = rp1**2 + (rp2/q1)**2 + (rp3/s1)**2 

    ind = np.where(rp <= (a**2/b/c)**(2/3) * app**2)
    #print( (a**2/b/c)**(2/3) * app**2)
    #ind = np.where(rp <= app**2)
    s = 1
    q = 1
    iter = 0
    while ((abs(1 - s1/s) >= 0.01) or (abs(1 - q1/q) >= 0.01 )) and (iter <= maxiter):
        iter += 1
        #print(a, b, c)
        s = s1
        q = q1 


        X1 = X[ind]
        Y1 = Y[ind]
        Z1 = Z[ind]
        mass1 = mass[ind]
        rp = rp[ind]

        X2 = wcow(X1, X1, mass1/rp)
        Y2 = wcow(Y1, Y1, mass1/rp)
        Z2 = wcow(Z1, Z1, mass1/rp)
        XY = wcow(X1, Y1, mass1/rp)
        XZ = wcow(X1, Z1, mass1/rp)
        YZ = wcow(Y1, Z1, mass1/rp)

        P = np.array([[X2, XY, XZ], [XY, Y2, YZ], [XZ, YZ, Z2]])
        lamb, vecs = eig(P)
        Jxx, Jyy, Jzz = lamb
        a = np.sqrt(Jxx)
        b = np.sqrt(Jyy)
        c = np.sqrt(Jzz)

        vx = vecs[:,0]
        vy = vecs[:,1]
        vz = vecs[:,2]

        axes = sorted([(a, vx), (b, vy), (c, vz)])

        a, ax_a = axes[2]
        b, ax_b = axes[1]
        c, ax_c = axes[0]
   
        s1 = c/a
        q1 = b/a
    
        rp1 = np.array([np.dot(ax_a, r) for r in zip(X, Y, Z)])
        rp2 = np.array([np.dot(ax_b, r) for r in zip(X, Y, Z)])
        rp3 = np.array([np.dot(ax_c, r) for r in zip(X, Y, Z)])

        rp = rp1**2 + (rp2/q1)**2 + (rp3/s1)**2 
        ind = np.where((rp >= (a**2/b/c)**(2/3) * app_min**2) * (rp <= (a**2/b/c)**(2/3) * app**2))
        
        #ind = np.where(rp <=  app**2)

        #print( (a**2/b/c)**(2/3) * app**2)

    
    #print(a, b , c)
    eps = 1 -  c/a
    T   = (a**2 - b**2) / (a**2 - c**2)
    

    #print(P@ax_a, a**2*ax_a)
    #print(P@ax_b, b**2*ax_b)
    #print(P@ax_c, c**2*ax_c)
    Q = solve(P)

    A = (a**2/b/c)**(1/3) * app 
    B = q1*(a**2/b/c)**(1/3) * app 
    C = s1*(a**2/b/c)**(1/3) * app 
    #assert iter < maxiter, 'Bad fit'
    if iter < maxiter:
        return A, B, C, Q, eps, T
    else: 
        return None


def shell_fit(coords : np.ndarray, mass : np.ndarray, velocity,  nbins : int = 20, ndots :int = None,
        rmin : float = 0, rmax :float = 0.03, method='numb', step='lin', II=None):
    
    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]

    r2 = np.sqrt(X**2 + Y**2 + Z**2)
    ind = np.where( (r2 <= rmax) * (r2 >= rmin))

    X1 = X[ind]
    Y1 = Y[ind]
    Z1 = Z[ind]
    mass1 = mass[ind]
    coords1 = coords[ind]
    velocity1 = velocity[ind]
    r2 = np.sqrt(X1**2 + Y1**2 + Z1**2)


    if method == 'numb':
        if ndots is None:
            bins = const_numb_bins(r2, nbins=nbins,verbous=False)
        else:
            bins = const_numb_bins(r2, nbins=nbins, ndots=ndots, verbous=False)
    else:
        bins = const_width_bins(r2, nbins=nbins, method=step)
    #print(bins)
    Psix_s = []
    Psiy_s = []
    Psiz_s = []

    Phix_s = []
    Phiy_s = []
    Phiz_s = []

    Thx_s = []
    Thy_s = []
    Thz_s = []

    ri    = []
    k = np.array([0, 0, 1])
    for i in range(len(bins) - 1):
        rmin_tmp = bins[i]
        rmax_tmp = bins[i+1]
        ind = np.where((r2 >= rmin_tmp) * (r2 <= rmax_tmp))
        Xi = X1[ind]
        Yi = Y1[ind]
        Zi = Z1[ind]
        Mi = mass1[ind]
        vi = velocity1[ind]  
        L = momentum(Xi, Yi, Zi, mass=Mi, velocity=vi, rmin=rmin_tmp, rmax=rmax_tmp)
        L = L/np.sqrt(np.sum(L**2))
        #print(L)
        Psix, Psiy, Psiz = L
        


        #try:
        #    A, B, C, Q, eps, T =  itter_tensor(Xi, Yi, Zi, Mi, app=0.03)
        #except:
        #    continue
        #ri.append((rmax_tmp + rmin_tmp)/2.)
        
        #Thx, Thy, Thz    = Q[:, 0]
        #Phix, Phiy, Phiz = Q[:, 1]
        #Psix, Psiy, Psiz = Q[:, 2]
        #print(A, B, C)
        
        ri.append((rmax_tmp + rmin_tmp)/2.)
        
        #Psi = np.degrees(np.arccos(np.dot(k, Q@k)))
        #if Psi > 90:
        #    Psi = 180 - Psi 
        #if Psiz < 0:
        #    Psiz = -Psiz
        #    Psix = - Psix
        #    Psiy = - Psiy

        #    Thx, Thy, Thz, Phix, Phiy, Phiz = Phix, Phiy, Phiz, Thx, Thy, Thz

        #Psix = np.degrees(np.arccos(Psix))
        #Psiy = np.degrees(np.arccos(Psiy))
        #Psiz = np.degrees(np.arccos(Psiz))

        #if Psix > 90:
        #    Psix = 180 - Psix

        #if Psiz > 90:
        #    Psiz = 180 - Psiz

        #if Psiy > 90:
        #    Psiy = 180 - Psiy

        Psix_s.append(Psix)
        Psiy_s.append(Psiy)
        Psiz_s.append(Psiz)

        #Phix_s.append(Phix)
        #Phiy_s.append(Phiy)
        #Phiz_s.append(Phiz)

        #Thx_s.append(Thx)
        #Thy_s.append(Thy)
        #Thz_s.append(Thz)

    ri = np.array(ri) 
    
    
    def foo1(x, r):
        arr = []
        x = np.array(x)
        n = len(x)
        for i in range(0, n, 1):
            for j in range(n-1, 0, -1):
                if  (abs(i - j) > 20) and(i < j):
                    x_tmp = x[i:j]
                    std = np.std(x_tmp)/len(x_tmp) 
                    arr.append([std, i, j])
        #print('arr', sorted(arr))
        _, i, j = sorted(arr)[0]
        #plt.figure()
        #plt.plot(range(n), x, '-ob')
        #plt.plot(range(n)[i:j], x[i:j], '-r')
        #plt.show()
        return i, j
    
    def multy_foo1(x, y, z, r):
        arr = []
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        n = len(x)
        for i in range(0, n, 1):
            for j in range(n-1, 0, -1):
                if  (abs(i - j) > 20) and(i < j):
                    x_tmp = x[i:j]
                    y_tmp = y[i:j]
                    z_tmp = y[i:j]
                    stdx = np.std(x_tmp)
                    stdy = np.std(y_tmp)
                    stdz = np.std(z_tmp)
                    std = np.sqrt(stdx**2 + stdy**2 + stdz**2)/np.log10(len(x_tmp))
                    arr.append([std, i, j])
        #print('arr', sorted(arr))
        _, i, j = sorted(arr)[0]
        return i, j
    #ix, jx = foo1(Psix_s, ri)
    #iy, jy = foo1(Psiy_s, ri)
    #iz, jz = foo1(Psiz_s, ri)

    #ii = np.min([ix, iy, iz])
    #jj = np.max([jx, jy, jz])

    ii, jj = multy_foo1(Psix_s, Psiy_s, Psiz_s, ri)

    Psix_s1 = Psix_s[ii:jj]
    Psiy_s1 = Psiy_s[ii:jj]
    Psiz_s1 = Psiz_s[ii:jj]
    ri1     = ri[ii:jj]

    def foo(Psix_s, Psiy_s, Psiz_s):
        Psix_s = np.array(Psix_s) 
        Psiy_s = np.array(Psiy_s) 
        Psiz_s = np.array(Psiz_s) 
        ll = np.sqrt(np.median(Psix_s)**2 + np.median(Psiy_s)**2 + np.median(Psiz_s)**2)
        a = np.median(Psix_s)/ll
        b = np.median(Psiy_s)/ll
        c = np.median(Psiz_s)/ll
        #print(a**2 + b**2 + c**2)
        return a, b, c
    
    Q1 = np.zeros((3,3))
    #Q1[:, 0] = foo(Thx_s, Thy_s, Thz_s)
    #Q1[:, 1] = foo(Phix_s, Phiy_s, Phiz_s)
    I = foo(Psix_s1, Psiy_s1, Psiz_s1)

    sint = I[2]
    cost = np.sqrt(1-sint**2)
    cosp = I[0]/cost
    sinp = I[1]/cost

    Rz = np.array([[cosp, sinp, 0], [-sinp, cosp, 0], [0, 0, 1]])
    Ry = np.array([[sint, 0, -cost], [0, 1, 0], [cost, 0, sint]])
    I1 = Ry @ (Rz @ I)
    #print(I, I1)
    Q1 = Ry @ Rz
    

    #print(np.dot(Q1[:,0], Q1[:,1]))
    #print(Q1.T @ Q1)
    a, b, c = I





    #from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #mm = 0.03
    #for ai, bi, ci in zip(Psix_s, Psiy_s, Psiz_s):
    #    ax.quiver(0, 0, 0, ai*mm, bi*mm, ci*mm, color='r')
    #ax.quiver(0, 0, 0, a*mm, b*mm, c*mm, color='g')
    #ax.scatter(X1, Y1, Z1, s=1, alpha=0.1)
    #plt.show()







    '''
    
    plt.figure()
    plt.plot((ri), Psix_s, '-o', color='grey', linewidth=1, label='Psi_x')
    plt.plot((ri1), Psix_s1, '-o', color='b', label='Flat_x')
    plt.gca().axhline(a, color='b', label='a')
    plt.gca().axhline(II[0], linestyle ='--', color='b', label='a')
    
    plt.plot((ri), Psiy_s, '-s', color='grey', label='Psi_y')
    plt.plot((ri1), Psiy_s1, '-sr', label='Flat_y')
    plt.gca().axhline(b, color='r' , label='b')
    plt.gca().axhline(II[1], linestyle ='--', color='r', label='a')


    plt.plot((ri), Psiz_s, '-^g', color='grey', label='Psi_z')
    plt.plot((ri1), Psiz_s1, '-^g', label='Flat_z')
    plt.gca().axhline(c, color='green', label='c')
    plt.gca().axhline(II[2], linestyle ='--', color='g', label='a')
    

    plt.legend()
    plt.xlabel('r, Mpc')
    plt.ylabel('Psi, deg')
    plt.show()
    '''   
    return Q1.T
    

def fit3D(coords, mass, niter=3, Amin=0):

    A = None
    B = None
    C = None
    Q = np.array([[1,0,0],[0,1,0],[0,0,1]])
    E1 ,E2, E3 = 0,0,0
    for _ in range(niter):
        #print(A, B, C)
        Q1, A, B, C = run(coords, mass, Q, A, B, C)
        #print(A, B, C)
        if A <= Amin:
            break
        else:
            Q = Q1
        #E1_, E2_, E3_ = anlge_from_matrix(Q)
        #print(E1_, E2_, E3_)
        #if (abs(E1_ - E1) < 1 and abs(E2_ - E2) < 1) and abs(E3_ - E3) < 5:
        #    break
        #else:
        #    E1 = E1_
        #    E2 = E2_
        #    E3 = E3_
    
    coords = np.array([Q.T @ a for a in coords])
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]

    ind = np.where(X**2/A**2 + Y**2/B**2 + Z**2/C**2 <= 1)
    coords1 =  coords[ind]

    X1 = coords1[:,0]
    Y1 = coords1[:,1]
    Z1 = coords1[:,2]

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = A * np.outer(np.cos(u), np.sin(v))
        y = B * np.outer(np.sin(u), np.sin(v))
        z = C * np.outer(np.ones(np.size(u)), np.cos(v))
        # Plot the surface
        theCM = mpl.colormaps.get_cmap('bwr')
        theCM._init()
        alphas = np.abs(np.linspace(-1.0, 1.0, theCM.N))
        theCM._lut[:-3,-1] = alphas

        # other code

        ax.plot_surface(x, y, z, cmap=theCM)
        ax.scatter(X, Y, Z, s=1, alpha=0.1, c='b')
        ax.scatter(X1, Y1, Z1, s=1, alpha=0.1, c='y')
        ax.set_xlim(-2*A, 2*A)
        ax.set_ylim(-2*A, 2*A)
        ax.set_zlim(-2*A, 2*A)


        
        #ax.scatter(X, Y, Z, s=1, c='b', alpha=0.1)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_aspect('equal')

        plt.show()
    return Q, A, B,  C#, coords, mass

def Deemer_method(star, gas, stallar_hmr, gas_hmr):

    g_coords = gas['coords']
    X = g_coords[:, 0]
    Y = g_coords[:, 1]
    Z = g_coords[:, 2]
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    ind = np.where(r < gas_hmr/2.)

    if False : #len(ind[0]) > 50:
        print('GAS')
        coord1 = g_coords[ind]
        vel1   = gas['velocity'][ind]
        mass1  = gas['mass'][ind]
        X1 = coord1[:, 0]
        Y1 = coord1[:, 1]
        Z1 = coord1[:, 2]
        L = momentum(X1, Y1, Z1, velocity=vel1, mass=mass1, rmin=-1, rmax=np.Infinity)
        I = L / np.sqrt(np.sum(L**2))
        print('I', I)
        sint = I[2]
        cost = np.sqrt(1-sint**2)
        cosp = I[0]/cost
        sinp = I[1]/cost

        Rz = np.array([[cosp, sinp, 0], [-sinp, cosp, 0], [0, 0, 1]])
        Ry = np.array([[sint, 0, -cost], [0, 1, 0], [cost, 0, sint]])
        I1 = Ry @ (Rz @ I)
        print(I, I1)
        Q1 = Ry @ Rz

    else:
        print('STAR')
        s_coords = star['coords']
        X = s_coords[:, 0]
        Y = s_coords[:, 1]
        Z = s_coords[:, 2]
        r = np.sqrt(X**2 + Y**2 + Z**2)
        ind = np.where(r < 2*stallar_hmr)

        coord1 = s_coords[ind]
        vel1   = star['velocity'][ind]
        mass1  = star['mass'][ind]
        X1 = coord1[:, 0]
        Y1 = coord1[:, 1]
        Z1 = coord1[:, 2]
        L = momentum(X1, Y1, Z1, velocity=vel1, mass=mass1, rmin=-1, rmax=np.Infinity)
        I = L / np.sqrt(np.sum(L**2))
        
        sint = I[2]
        cost = np.sqrt(1-sint**2)
        cosp = I[0]/cost
        sinp = I[1]/cost

        Rz = np.array([[cosp, sinp, 0], [-sinp, cosp, 0], [0, 0, 1]])
        Ry = np.array([[sint, 0, -cost], [0, 1, 0], [cost, 0, sint]])
        I1 = Ry @ (Rz @ I)
        print(I, I1)
        Q1 = Ry @ Rz
        
    return Q1.T



if __name__ == '__main__':

    for _ in range(10):
        a = np.random.random()*90
        b = np.random.random()*90
        c = np.random.random()*90
        print('abc', a, b, c)
        coord, mass = Gauss3D(100_000, 100, 100, 30, a, b, c)
        from check_orient import * 
       
        tx, ty, R = check2D(coord, mass, 30, id, kco=0, M=0, A=100)
        print(a, b, c)
        print(tx, ty)
        #fit3D(coord, mass)
        break

def tensorI(coord :np.ndarray, velocity :np.ndarray, mass :np.ndarray, 
            rmin :float =0. , rmax : float =0.02):
    r2 = np.einsum('...j,...j->...',coord,coord)
    ind = np.where(((r2 > rmin**2) * (r2 < rmax**2)))

    coord1    = coord[ind]
    velocity = velocity[ind]
    mass     = mass[ind]

    x = coord1[:, 0]
    y = coord1[:, 1]
    z = coord1[:, 2]

    Ixx = np.sum((y**2 + z**2)*mass)
    Iyy = np.sum((x**2 + z**2)*mass)
    Izz = np.sum((x**2 + y**2)*mass)

    Ixy = np.sum(x*y*mass)
    Iyz = np.sum(z*y*mass)
    Ixz = np.sum(x*z*mass)

    T = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])
    #print('Tensor')
    #print(T)
    from numpy.linalg import eig

    lambs, vecs = eig(T)

    J = vecs.transpose() @ T @ vecs
    Jxx = J[0,0]
    Jyy = J[1,1]
    Jzz = J[2,2]

    if (Jzz > Jyy) and (Jzz > Jxx):
        Q = vecs
        
    elif (Jxx > Jyy) and (Jxx > Jzz):
        R = np.array([[0,0,1], [0, 1, 0], [-1, 0, 0]])
        Q = vecs @ R
        #print(R.transpose() @ T @ R)
    else:
        R = np.array([[1,0,0], [0, 0, 1], [0, -1, 0]])
        Q =  vecs @ R
        #print(R.transpose() @ T @ R)
    J = Q.transpose() @ T @ Q
    Jxx = J[0,0]
    Jyy = J[1,1]
    Jzz = J[2,2]
    print('CAB', np.sqrt(1/Jzz)*3, np.sqrt(1/Jxx)*3, np.sqrt(1/Jyy)*3)

    return Q,  np.sqrt(1/Jzz)*3, np.sqrt(1/Jxx)*3, np.sqrt(1/Jyy)*3


    '''
    def momentum(coord :np.ndarray, velocity :np.ndarray, mass :np.ndarray, 
                rmin :float =0.0025, rmax :float =  0.03):
        I = np.zeros(3)
        
        x = coord[:,0]
        y = coord[:,1]
        z = coord[:,2]
        d = np.array(np.sqrt(x**2 + y**2 + z**2))
        #d = np.array( [(xi/A)**2 + (yi/B)**2 + (zi/C)**2 for xi, yi, zi in zip(x, y,z)])
        
        ind = np.where((d >= rmin ) * (d <= rmax))
        
        coord1    = coord[ind]
        velocity1 = velocity[ind]
        mass1     = mass[ind]
        for r, v, m in zip(coord1, velocity1, mass1):
            I += m * np.cross(r, v)
        return I 

    def momentum_ell(coord :np.ndarray, velocity :np.ndarray, mass :np.ndarray,
                    A : float, B :float, C :float, rmin :float, rmax :float):
        I = np.zeros(3)
        
        x = coord[:,0]
        y = coord[:,1]
        z = coord[:,2]
        #d = np.array(np.sqrt(x**2 + y**2 + z**2))
        d = np.array( [(xi/A)**2 + (yi/B)**2 + (zi/C)**2 for xi, yi, zi in zip(x, y,z)])
        
        k = (rmin/rmax)**2
        ind = np.where((d >= k) * (d <= 1))
        
        coord1    = coord[ind]
        velocity1 = velocity[ind]
        mass1     = mass[ind]
        for r, v, m in zip(coord1, velocity1, mass1):
            I += m * np.cross(r, v)/abs(r)
        return I 

    def momentum_fit(coords :np.ndarray, velocity :np.ndarray, mass :np.ndarray, 
                    rmin :float =0.0025 , rmax :float =0.03 ):
       
        #    Rotate data according angular momentum vector
        
        L = momentum(coords, velocity, mass, rmin=rmin, rmax=rmax)
        I = L/np.sqrt(np.sum(L**2))
        print('I', I)
            
        sint = I[2]
        cost = np.sqrt(1-sint**2)
        cosp = I[0]/cost
        sinp = I[1]/cost

        Rz = np.array([[cosp, sinp, 0], [-sinp, cosp, 0], [0, 0, 1]])
        Ry = np.array([[sint, 0, -cost], [0, 1, 0], [cost, 0, sint]])
        I1 = Ry @ (Rz @ I)
        print(I1)
        Q = Ry @ Rz
        return Q.T
    '''