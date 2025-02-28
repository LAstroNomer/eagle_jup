# This function determinate 3D galaxy orintation
# by calculation first and second mass momentum

# import packages
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from numpy.linalg import eig

def first_momentum(x, m):
    return np.sum(x*m)/np.sum(m)

def second_momentum(x, m, xc):
    return np.sum(m*x**2)/np.sum(m) - xc**2

def cross_momentum(x, y, m, xc, yc):
    return np.sum(x*y*m)/np.sum(m) - xc*yc

def tensor(coord, mass, rmin=0, rmax=0.02):

    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]

    Ixx = np.sum((y**2 + z**2)*mass)
    Iyy = np.sum((x**2 + z**2)*mass)
    Izz = np.sum((x**2 + y**2)*mass)

    Ixy = np.sum(x*y*mass)
    Iyz = np.sum(z*y*mass)
    Ixz = np.sum(x*z*mass)

    T = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])
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

    J = Q.T @ T @ Q

    #print(J)

    Ix = J[0,0]
    Iy = J[1,1]
    Iz = J[2,2]

    A = 1/np.sqrt(Ix)
    B = 1/np.sqrt(Iy)
    C = 1/np.sqrt(Iz)

    #print(A, B, C)
    return Q, A, B, C

def fit2D(X, Y, mass):

    xc = first_momentum(X, mass)
    yc = first_momentum(Y, mass)
   

    X2 = second_momentum(X, mass, xc)
    Y2 = second_momentum(Y, mass, yc)
    XY = cross_momentum(X, Y, mass, xc, yc)


    theta = np.arctan(2*XY/(X2-Y2))/2
    A = np.sqrt((X2 + Y2)/2 +np.sqrt(((X2 - Y2)/2)**2 + XY**2))
    B = np.sqrt((X2 + Y2)/2 - np.sqrt(((X2 - Y2)/2)**2 + XY**2))

    if B > A:
        return xc, yc, B, A, theta+np.pi/2 

    return xc, yc, A, B, theta 

def fit3D(X, Y, Z, mass):
    xc = first_momentum(X, mass)
    yc = first_momentum(Y, mass)
    zc = first_momentum(Z, mass)
   

    X2 = second_momentum(X, mass, xc)
    Y2 = second_momentum(Y, mass, yc)
    Z2 = second_momentum(Z, mass, zc)
    XY = cross_momentum(X, Y, mass, xc, yc)
    XZ = cross_momentum(X, Z, mass, xc, zc)
    YZ = cross_momentum(Y, Z, mass, yc, zc)

    beta = np.arctan(2*YZ/(Z2-Y2))/2
    cb = np.sin(beta)
    sb = np.sin(beta)

    alpha = np.arctan((2*(XY*cb - XZ*sb))/(Y2*cb**2 + Z2*sb**2 + -X2 - 2*YZ*sb*cb))/2
    sa = np.sin(alpha)
    ca = np.cos(alpha)

    A = np.sqrt(Z2*sa**2*sb**2 + (-(2*YZ*sa**2*cb)+2*XZ*ca*sa)*sb + Y2*sa**2*cb**2 - 2 *XY*ca*sa*cb+X2*ca**2)
    B = np.sqrt(Z2*ca**2*sb**2+(-(2*YZ*ca**2*cb)+(2*YZ-2*XZ)*ca*sa)*sb+Y2*ca**2*cb**2+(-(2*Y2)+2*XY)*ca*sa*cb+(Y2-2*XY+X2)*sa**2)
    C = np.sqrt(Y2*sb**2+2*YZ*cb*sb+Z2*cb**2)
    #print(alpha, beta, A, B, C)
    return alpha, beta, A, B, C

def fit3D_new(X, Y, Z, mass):
    

    
    xc = first_momentum(X, mass)
    yc = first_momentum(Y, mass)
    zc = first_momentum(Z, mass)
   

    X2 = second_momentum(X, mass, xc)
    Y2 = second_momentum(Y, mass, yc)
    Z2 = second_momentum(Z, mass, zc)
    XY = cross_momentum(X, Y, mass, xc, yc)
    XZ = cross_momentum(X, Z, mass, xc, zc)
    YZ = cross_momentum(Y, Z, mass, yc, zc)

    P = np.array([[X2, XY, XZ], [XY, Y2, YZ], [XZ, YZ, Z2]])




    dots = [a @ P @ a for a in zip(X, Y, Z)]
    d = np.sqrt(np.sum(dots))
    print(d)
    plt.figure()
    plt.hist(dots)
    plt.show()
    lambs, Q = eig(P)

    A, B, C = 1/np.sqrt(lambs) * d

    print(A, B, C)
    return A, B, C, Q



def main(coords, mass, rmax):

    #plt.figure()
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]



    
    #plt.scatter(X, Y, s=1)
    xc, yc, A, B, theta = fit2D(X, Y, mass)
    #print(A, B)
    #lpha, beta, A, B, C = fit3D(X, Y, Z, mass)

    A, B, C, Q = fit3D_new(X, Y, Z, mass)
    
    coords = np.array([Q.transpose() @i for i in coords])
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2]



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
    ax.scatter(X, Y, Z, s=1, alpha=0.1)
    #ax.set_xlim(-2*A, 2*A)
    #ax.set_ylim(-2*A, 2*A)
    #ax.set_zlim(-2*A, 2*A)


    
    #ax.scatter(X, Y, Z, s=1, c='b', alpha=0.1)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()



    '''
    
    t = np.arange(0, 2*np.pi, 0.001)
    xt = A*np.cos(t)
    yt = B*np.sin(t)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    print('theta', np.degrees(theta))
    xtr = [(R@a)[0] for a in zip(xt, yt)]
    ytr = [(R@a)[1] for a in zip(xt, yt)]
    plt.plot(xc + xtr, yc + ytr, '-r')
    plt.plot(xc + xt, yc + yt, '-g')
    plt.show()
    '''













    '''
    #Q, A, B, C = tensor(coords, mass, rmax=rmax)

    #coords = np.array([Q.transpose() @i for i in coords])

    nbin = 200
    fig = plt.figure()
    fdata, _, _, _ = plt.hist2d(coords[:,0], coords[:,1], weights=mass, bins=nbin, range=[[-0.02, 0.02], [-0.02, 0.02]],norm=mcolors.LogNorm(), cmin=1)
    edata, _, _, _ = plt.hist2d(coords[:,0], coords[:,2], weights=mass, bins=nbin, range=[[-0.02, 0.02], [-0.02, 0.02]],norm=mcolors.LogNorm(), cmin=1)
    plt.close(fig)

    h1, w1 = fdata.shape
    pc2pix = 40000/h1
    fdata = fdata/pc2pix**2
    fdata[np.where(np.isnan(fdata))] = 1
   
    h1, w1 = edata.shape
    pc2pix = 40000/h1
    edata = edata/pc2pix**2
    edata[np.where(np.isnan(edata))] = 1
   
    t = np.arange(0, 2*np.pi, 0.01)
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(fdata, norm=mcolors.LogNorm(), origin='lower')
    ax[1].imshow(edata, norm=mcolors.LogNorm(), origin='lower')
    ax[0].plot(nbin/2 + A*np.cos(t), nbin/2+B*np.sin(t), '--', color='pink')
    plt.show()
    '''
    return

if __name__ == '__main__':
    pass