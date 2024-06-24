from scipy.special import jv, yv
import numpy as np

def mie_pt(u,nmax):
    u = float(u)
    p = np.zeros(nmax, dtype=float)
    p[0] = 1
    p[1] = 3*u
    t = np.zeros(nmax, dtype=float)
    t[0] = u
    t[1] = 6*u**2 - 3

    nn = np.arange(2,nmax,dtype=float)
    for n in nn:
        n_i = int(n)
        p[n_i] = (2*n+1)/n*p[n_i-1]*u - (n+1)/n*p[n_i-2]

    t[2:] = (nn+1)*u*p[2:] - (nn+2)*p[1:-1]

    return (p,t)

def mie_coeffs(params, u):
    '''Input validation and function selection for the Mie coefficients.
    '''

    eps = complex(params['eps']) if params['eps'] is not None else None
    x = float(params['x']) if params['x'] is not None else None

    if (x==None) or (eps==None):
        raise ValueError('Must specify x and either eps or m.')

    mu = 1.0

    y = float(params['y']) if params['y'] is not None else None
    eps2 = complex(params['eps2']) if params['eps2'] is not None else None

    coated = (y is not None)
    if coated == (eps2 is None):
        raise ValueError('Must specify both y and m2 for coated particles.')

    if not coated:
        y = x
        eps2 = eps

    # Do not use the coated version if it is not necessary
    if x==y or eps==eps2:
        coeffs = single_mie_coeff(eps,mu,y)
    elif x==0:
        coeffs = single_mie_coeff(eps2,mu,y)
    else:
        coeffs = coated_mie_coeff(eps,eps2,x,y)

    return mie_S12(coeffs, u)

def single_mie_coeff(eps,mu,x):
    """Mie coefficients for the single-layered sphere.

    Args:
        eps: The complex relative permittivity.
        mu: The complex relative permeability.
        x: The size parameter.

    Returns:
        A tuple containing (an, bn, nmax) where an and bn are the Mie
        coefficients and nmax is the maximum number of coefficients.
    """
    z = np.sqrt(eps*mu)*x
    m = np.sqrt(eps/mu)

    nmax = int(round(2+x+4*x**(1.0/3.0)))
    nmax1 = nmax-1
    nmx = int(round(max(nmax,abs(z))+16))
    n = np.arange(nmax)
    nu = n+1.5

    sx = np.sqrt(0.5*np.pi*x)
    px = sx*jv(nu,x)
    p1x = np.hstack((np.sin(x), px[:nmax1]))
    chx = -sx*yv(nu,x)
    ch1x = np.hstack((np.cos(x), chx[:nmax1]))
    gsx = px-complex(0,1)*chx
    gs1x = p1x-complex(0,1)*ch1x

    dnx = np.zeros(nmx,dtype=complex)
    for j in range(nmx-1,0,-1):
        r = (j+1.0)/z
        dnx[j-1] = r - 1.0/(dnx[j]+r)
    dn = dnx[:nmax]
    n1 = n+1
    da = dn/m + n1/x
    db = dn*m + n1/x

    an = (da*px-p1x)/(da*gsx-gs1x)
    bn = (db*px-p1x)/(db*gsx-gs1x)

    return (an, bn, nmax)

def coated_mie_coeff(eps1,eps2,x,y):
    """Mie coefficients for the dual-layered (coated) sphere.

       Args:
          eps: The complex relative permittivity of the core.
          eps2: The complex relative permittivity of the shell.
          x: The size parameter of the core.
          y: The size parameter of the shell.

       Returns:
          A tuple containing (an, bn, nmax) where an and bn are the Mie
          coefficients and nmax is the maximum number of coefficients.
    """
    m1 = np.sqrt(eps1)
    m2 = np.sqrt(eps2)
    m = m2/m1
    u = m1*x
    v = m2*x
    w = m2*y

    nmax = int(round(2+y+4*y**(1.0/3.0)))
    mx = max(abs(m1*y),abs(w))
    nmx = int(round(max(nmax,mx)+16))
    nmax1 = nmax-1
    n = np.arange(nmax)

    dnu = np.zeros(nmax,dtype=complex)
    dnv = np.zeros(nmax,dtype=complex)
    dnw = np.zeros(nmax,dtype=complex)
    dnx = np.zeros(nmx,dtype=complex)

    for (z, dn) in zip((u,v,w),(dnu,dnv,dnw)):
        for j in range(nmx-1,0,-1):
            r = (j+1.0)/z
            dnx[j-1] = r - 1.0/(dnx[j]+r)
        dn[:] = dnx[:nmax]

    nu = n+1.5
    vwy = [v,w,y]
    sx = [np.sqrt(0.5*np.pi*xx) for xx in vwy]
    (pv,pw,py) = [s*jv(nu,xx) for (s,xx) in zip(sx,vwy)]
    (chv,chw,chy) = [-s*yv(nu,xx) for (s,xx) in zip(sx,vwy)]
    p1y = np.hstack((np.sin(y), py[:nmax1]))
    ch1y = np.hstack((np.cos(y), chy[:nmax1]))
    gsy = py-complex(0,1)*chy
    gs1y = p1y-complex(0,1)*ch1y

    uu = m*dnu-dnv
    vv = dnu/m-dnv
    fv = pv/chv
    #fw = pw/chw
    ku1 = uu*fv/pw
    kv1 = vv*fv/pw
    pt = pw-chw*fv
    prat = pw/pv/chv
    ku2 = uu*pt+prat
    kv2 = vv*pt+prat
    dns1 = ku1/ku2
    gns1 = kv1/kv2

    dns = dns1+dnw
    gns = gns1+dnw
    nrat = (n+1)/y
    a1 = dns/m2+nrat
    b1 = m2*gns+nrat
    an = (py*a1-p1y)/(gsy*a1-gs1y)
    bn = (py*b1-p1y)/(gsy*b1-gs1y)

    return (an, bn, nmax)

def mie_S12(coeffs,u):
    """The amplitude scattering matrix.
    """
    (pin,tin) = mie_pt(u,coeffs[2])
    n = np.arange(1, coeffs[2]+1, dtype=float)
    n2 = (2*n+1)/(n*(n+1))
    pin *= n2
    tin *= n2
    return np.dot(coeffs[0],pin)+np.dot(coeffs[1],tin)

def mie_coated(r, t, n_s, n_i, n_m, l):
    params = {'x': (2*np.pi*n_m/l)*(r-t), 'y': (2*np.pi*n_m/l)*r,  \
               'eps': (n_i/n_m)**2, 'eps2': (n_s/n_m)**2}
    return np.abs(mie_coeffs(params, np.cos(np.pi/2)))**2
