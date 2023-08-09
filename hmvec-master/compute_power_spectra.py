import hmvec as hm
import numpy as np
import scipy as scp
from scipy.special import eval_legendre, legendre, spherical_jn
import itertools
import wigner
import time
from scipy import interpolate
from itertools import cycle
from math import atan2,degrees,lgamma 

from params import *
#from plotting import *

############### COMPUTE ANGULAR POWER SPECTRA ###########################

# Compute crossing radius of the Milky Way
def dark_photon_conv_prob_MilkyWay_gas(mMWvir, rMWvir, rhocritzMW, deltavMW, csMW, HMW, rE, rs, mDP, name='battagliaAGN'):
    delta_rhos1 = rhocritzMW*deltavMW
    delta_rhos2 = 200.*rhocritzMW
    m200critzMW = hm.mdelta_from_mdelta_unvectorized(mMWvir, csMW, delta_rhos1, delta_rhos2)
    r200critzMW = hm.R_from_M(m200critzMW, rhocritzMW, delta=200.)

    gas_profile = get_gas_profile(rs, 0., m200critzMW, r200critzMW, rhocritzMW, name=name)

    if mDP**2. < np.max(conv*gas_profile):
        idx  = np.argmin(np.abs(conv*gas_profile - mDP**2.))
        rcross = rs[idx]

        dmdr = np.abs(conv*get_deriv_gas_profile(rcross, 0., m200critzMW, r200critzMW, rhocritzMW, name=name))

        limits = np.heaviside(rMWvir-rcross, 1)#*np.heaviside(rcross-rE, 1.)

        prob = np.pi*mpcEVinv*(mDP**4.)*limits/dmdr
    else:
        rcross, prob = np.nan, 0.
    return rcross, prob

def dark_photon_conv_prob_MilkyWay_NFW(mMWvir, rMWvir, csMW, HMW, rE, rs, mDP):
    rss = rMWvir/csMW
    zMW = 0.

    nfw_rhoscsales = hm.rhoscale_nfw(mMWvir, rMWvir, csMW)
    nfw_profiles   = hm.rho_nfw(rs, nfw_rhoscsales, rss)

    idx  = np.argmin(np.abs(conv*nfw_profiles - mDP**2.))
    rcross = rs[idx]

    rfr  = rcross/rss
    dmdr = np.abs(conv*(nfw_rhoscsales/rss)*(1.+3.*rfr)/(rfr)**2./(1.+rfr)**3.)
    limits = np.heaviside(rMWvir-rcross, 1)#*np.heaviside(rcross-rE, 1.)

    prob = np.pi*mpcEVinv*(mDP**4.)*limits/(1.+zMW)/dmdr
    return rcross, prob

def get_volume_conv(chis, Hz):
    # Volume of redshift bin divided by Hubble volume
    # Chain rule factor when converting from integral over chi to integral over z
    return chis**2. / Hz

def get_rcross_per_halo_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, name='battagliaAGN'):
    # Compute crossing radius of each halo
    # i.e. radius where plasma mass^2 = dark photon mass^2
    NZ, NM = len(zs), len(ms)

    rcross = np.zeros((NZ,NM))

    m200critz, r200critz = get_200critz(zs, ms, cs, rhocritz, deltav)
    for zi, zz in enumerate(zs):
        for mi, mm in enumerate(ms):
            gas_profile = get_gas_profile(rs, zz, m200critz[zi,mi], r200critz[zi,mi], rhocritz[zi], name=name)
            # Find the index of the radius array where plasmon mass^2 = dark photon mass^2
            if mDP**2. < np.max(conv*gas_profile):
                idx = np.argmin(np.abs(conv*gas_profile - mDP**2.))
                # Get radius in each halo where this resonance happens
                rcross[zi, mi] = rs[idx]
            else:
                rcross[zi, mi] = np.nan
    return rcross

def get_200critz(zs, ms, cs, rhocritz, deltav):
    delta_rhos1 = rhocritz*deltav
    delta_rhos2 = 200.*rhocritz
    m200critz = hm.mdelta_from_mdelta(ms, cs, delta_rhos1, delta_rhos2)
    r200critz = hm.R_from_M(m200critz, rhocritz[:,None], delta=200.)
    return m200critz, r200critz

def get_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    #choose profile
    if name=='ACT':
        rho0, alpha, beta, gamma, xc = bestfitACT()
    elif name=='battagliaSH':
        rho0, alpha, beta, gamma, xc = battagliaSH(m200, zs)
    else:
        rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    rho = rhocritz * rho0
    x = rs/r200/xc
    expo = -(beta+gamma)/alpha     # gamma sign must be opposite from Battaglia/ACT paper; typo
    rhogas = rho * (x**gamma) * ((1. + x**alpha)**expo)
    return rhogas

def get_rcross_per_halo_NFW(zs, ms, rs, rvir, cs, mDP):
    # Compute crossing radius of each halo
    # i.e. radius where plasma mass^2 = dark photon mass^2
    NZ, NM = len(zs), len(ms)

    rcross = np.zeros((NZ,NM))
    for zi, zz in enumerate(zs):
        rss = rvir[zi]/cs[zi]

        rhos = hm.rhoscale_nfw(ms, rvir[zi], cs[zi])
        nfw  = np.asarray([hm.rho_nfw(rs, rhos[mi], rss[mi]) for mi in range(NM)])

        # Find the index of the radius array where plasmon mass^2 = dark photon mass^2
        idx = np.argmin(np.abs(conv*nfw - mDP**2.), axis=1)

        # Get radius in each halo where this resonance happens
        rcross[zi] = rs[idx]
    return rcross

def get_halo_skyprofile(zs, chis, rcross):
    # get bounds of each regime within halo
    rchis   = chis*aa(zs)
    rchis2D = np.outer(rchis, np.ones(rcross.shape[-1]))
    
    rcross[np.isnan(rcross)] = 0.

    listincr = 1. - np.geomspace(1e-5, 1., 40)
    listincr = np.asarray([1.] + listincr.tolist())[::-1]
    angs = np.multiply.outer(listincr, rcross/rchis2D)
    if np.all(angs.flatten()==0.):
        ucosth = np.ones(np.shape(angs))
    else:
        ucosth = (1.-(angs * rchis2D/rcross)**2.)**(-0.5)
        ucosth[angs==rcross/rchis2D] = 0.
    return ucosth, angs

def get_halo_skyprofile2(zs, chis, rcross, play):
    # get bounds of each regime within halo
    rchis   = chis*aa(zs)
    rchis2D = np.outer(rchis, np.ones(rcross.shape[-1]))

    rcross[np.isnan(rcross)] = 0.

    listincr = 1. - np.geomspace(play, 1., 40)
    listincr = np.asarray([1.] + listincr.tolist())[::-1]
    angs = np.multiply.outer(listincr, rcross/rchis2D)
    if np.all(angs.flatten()==0.):
        ucosth = np.ones(np.shape(angs))
    else:
        ucosth = (1.-(angs * rchis2D/rcross)**2.)**(-0.5)
    return ucosth, angs


def get_u00(angs, ucosth):
    # angular function u(theta) projected into multipoles
    cos_angs = np.cos(angs) # sin(angs) \sim angs
    P0  = eval_legendre(0., cos_angs)
    u00 = 2.*np.pi * np.trapz(angs * ucosth * P0, angs, axis=0)
    return u00 / (4.*np.pi)

def get_uell0(angs, ucosth, ell):
    # angular function u(theta) projected into multipoles
    cos_angs  = np.cos(angs)
    sin_angs  = np.sin(angs)
    Pell      = eval_legendre(ell, cos_angs)
    uL0       = 2.*np.pi * np.trapz(sin_angs * ucosth * Pell, angs, axis=0)

    if ell%100==0: print(ell)
    return uL0 * (4.*np.pi / (2.*ell+1.))**(-0.5)

def dark_photon_conv_prob_NFW(zs, ms, rs, rvir, cs, mDP, rcross, rscale=False):
    NZ, NM = len(zs), len(ms)
    prob   = np.zeros((NZ,NM)) # Array for optical depth per halo due to A' conversion.

    for zi, zz in enumerate(zs):
        rss  = rvir[zi]/cs[zi]
        rhos = hm.rhoscale_nfw(ms, rvir[zi], cs[zi])

        # d m_gamma^2 / dr evaluated where plasma mass^2 = A' mass^2
        rfr  = rcross[zi]/rss
        dmdr = np.abs(conv*(rhos/rss)*(1.+3.*rfr)/(rfr)**2./(1.+rfr)**3.)

        # Conversion probability per halo, ensure (rcross < r_vir) with Heaviside function
        # The factor of 2 comes from the fact that if conversion does happen it happens in + out
        # unless it happens exactly at rcross = r_vir; in this case we divide by 2, hence the normalization of theta
        omegaz = (1.+zz)#*omega0 it doesn't change the phenomenology, but we want to remove frequency dependence later.
        if rscale:
            uang = 2.*np.heaviside(rvir[zi]-rcross[zi], 0.5)*np.heaviside(rcross[zi]-rss, 1.)
        else:
            uang = 2.*np.heaviside(rvir[zi]-rcross[zi], 0.5)

        prob[zi] = np.pi * mpcEVinv * (mDP**4.) * uang / omegaz / dmdr
        prob[zi][np.isnan(rcross[zi])] = 0.
    return prob

def get_deriv_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    #choose profile
    if name=='ACT':
        rho0, alpha, beta, gamma, xc = bestfitACT()
    elif name=='battagliaSH':
        rho0, alpha, beta, gamma, xc = battagliaSH(m200, zs)
    else:
        rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    x = rs / r200 / xc
    P = rhocritz * rho0
    expo = -(alpha+beta+gamma)/alpha
    drhodr = P * (x**gamma) * (1. + x**alpha)**expo * (gamma - x**alpha * beta) / rs
    if hasattr(rs, "__len__"):
        drhodr[rs==0.] = 0.
    elif rs==0:
        drhodr = 0.
    return drhodr


def get_halo_profile_vir(rchi, rmin, rvir):
    l1 = 1e-5 + 1. - np.geomspace(1., 1e-5, 20)
    l7 = np.geomspace(1e-5, 1., 20)
    listincr = np.sort(np.concatenate(([rmin],l1, l7), axis=0))*rvir/rchi
    return np.asarray(listincr)

def get_drdt(angth, rchi, rvir, rs):
    ratio = angth*rchi/rs
    drdt = (1.-ratio**2.)**-0.5
    drdt[ratio>1.] = 0.
    return drdt

def get_thomsontau_per_halo_old(zs, ms, rs, chis, rvirs, rhocritz, deltav, cs, mDP, name='battagliaAGN'):
    NZ, NM = len(zs), len(ms)
    rhohalo = np.empty((NZ, NM))
    ucosth = np.empty((41, NZ, NM))
    angsave = np.empty((41, NZ, NM))

    m200critz, r200critz = get_200critz(zs, ms, cs, rhocritz, deltav)
    rmin = min(rs)
    for zi, zz in enumerate(zs):
        rchi = chis[zi]*aa(zz)
        rhoc = rhocritz[zi]

        for mi, mm in enumerate(ms):
            m200 = m200critz[zi,mi]
            r200 = r200critz[zi,mi]
            rvir = rvirs[zi,mi]

            coords1 = (rs>=rvir)
            coords2 = (rs<rvir)

            angs = get_halo_profile_vir(rchi, rmin, rvir)
            gas_profile = get_gas_profile(rs, zz, m200, r200, rhoc, name=name)
            gas_profile[coords1] = 0.

            semiprof = np.empty(len(angs))
            for th, theta in enumerate(angs):
                dtdr = np.zeros(len(rs))
                dtdr[coords2] = get_drdt(theta, rchi, rvir, rs[coords2])
                semiprof[th]  = np.trapz(gas_profile*dtdr*rs**2., rs, axis=0)

            rhohalo[zi,mi]   = 2.*np.pi * np.trapz(np.sin(angs) * semiprof, angs, axis=0) # 2 * pi * d theta sin theta
            ucosth[:,zi,mi]  = semiprof/rhohalo[zi,mi]
            angsave[:,zi,mi] = angs
            rhohalo[zi,mi]   = 2.*conv2 * aa(zz) * rhohalo[zi,mi]
    return rhohalo, ucosth, angsave # 2 from in and out

def get_thomsontau_per_halo(zs, ms, ellMax, chis, rvirs, rhocritz, deltav, cs, mDP, name='battagliaAGN'):
    NZ, NM = len(zs), len(ms)
    rhohalo = np.zeros((ellMax, NZ, NM))
    ells = np.arange(ellMax)

    m200critz, r200critz = get_200critz(zs, ms, cs, rhocritz, deltav)

    rit1 = 1e-10 + 1. - np.geomspace(1., 1e-10, 10000)
    rit2 = np.geomspace(1e-10, 1., 10000)
    rits = np.sort(np.concatenate((rit1,rit2), axis=0))

    for zi, zz in enumerate(zs):
        rchi = chis[zi]*aa(zz)
        rhoc = rhocritz[zi]
        multp = 4.*np.pi * conv2 * aa(zz)/chis[zi]**2.

        for mi, mm in enumerate(ms):
            m200 = m200critz[zi,mi]
            r200 = r200critz[zi,mi]
            rvir = rvirs[zi,mi]

            rs = rits*rvir
            gas_profile = get_gas_profile(rs, zz, m200, r200, rhoc, name=name)

            for li, ell in enumerate(ells):
                fact1 = (ell + 0.5)*rs/chis[zi]
                rhohalo[li,zi,mi] = multp * np.trapz(np.sin(fact1)/fact1 * gas_profile * rs**2., rs, axis=0)
    return rhohalo

def dark_photon_conv_prob_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, name='battagliaAGN'):
    NZ, NM = len(zs), len(ms)
    m200critz, r200critz = get_200critz(zs, ms, cs, rhocritz, deltav)

    prob   = np.zeros((NZ,NM)) # Array for optical depth per halo due to A' conversion.
    for zi, zz in enumerate(zs):
        # d m_gamma^2 / dr evaluated where plasma mass^2 = A' mass^2
        m200 = m200critz[zi,:]
        r200 = r200critz[zi,:]
        rhocr = rhocritz[zi]
        drhodr = get_deriv_gas_profile(rcross[zi], zz, m200, r200, rhocr, name=name)
        dmdr = np.abs(conv*drhodr)
        omegaz = (1.+zz)#*omega0 it doesn't change the phenomenology, but we want to remove frequency dependence later.
        uang = 2.*np.heaviside(rvir[zi]-rcross[zi], 0.5)

        prob[zi] = np.pi * mpcEVinv * (mDP**4.) * uang / omegaz / dmdr
        prob[zi][np.isnan(rcross[zi])] = 0.
    return prob

def get_avtau(zs, ms, nzm, prob, dvol, u00):
    # Average optical depth per redshift bin
    dtaudz = np.trapz(nzm * prob * u00, ms, axis=-1) * dvol * 4*np.pi
    avtau  = np.trapz(dtaudz, zs, axis=0)
    return avtau, dtaudz

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def get_Celldtaudtau_1h(zs, ms, ks, nzm, dvol, Prob, uell0, ells):
    # The 1-halo term
    Cl1h = np.trapz(nzm * np.abs(Prob * uell0)**2., ms, axis=-1)

    for ell in range(np.shape(Cl1h)[0]):
        nans, x = nan_helper(Cl1h[ell])
        Cl1h[ell][nans]= np.interp(x(nans), x(~nans), Cl1h[ell][~nans])
 #       print('1h <<nan>> redshift indices:', nans, 'at ell', ell)

    CldTau1h = np.trapz(Cl1h * dvol, zs, axis=-1)
    return CldTau1h * (4.*np.pi) / (2.*ells+1.)

def get_Celldtaudthom_1h(zs, ms, ks, nzm, dvol, thom_probell, Prob, uell0, ells):
    # The 1-halo term
    Cl1h = np.trapz(nzm * thom_probell * (Prob * uell0), ms, axis=-1)

    for ell in range(np.shape(Cl1h)[0]):
        nans, x = nan_helper(Cl1h[ell])
        Cl1h[ell][nans]= np.interp(x(nans), x(~nans), Cl1h[ell][~nans])
 #       print('1h <<nan>> redshift indices:', nans, 'at ell', ell)

    CldTau1h = np.trapz(Cl1h * dvol, zs, axis=-1)
    return CldTau1h * ((4.*np.pi) / (2.*ells+1.))**0.5

def get_Celldthomdthom_1h(zs, ms, ks, nzm, dvol, probell, ells):
    # The 1-halo term
    Cl1h = np.trapz(nzm * np.abs(probell)**2., ms, axis=-1)

    for ell in range(np.shape(Cl1h)[0]):
        nans, x = nan_helper(Cl1h[ell])
        Cl1h[ell][nans]= np.interp(x(nans), x(~nans), Cl1h[ell][~nans])
 #       print('1h <<nan>> redshift indices:', nans, 'at ell', ell)

    CldTau1h = np.trapz(Cl1h * dvol, zs, axis=-1)
    return CldTau1h

def get_dtauell(ms, nzm, Prob, dvol, biases, uell0):
    # integrand in 2-halo numerator
    return np.trapz(biases * nzm * Prob * uell0, ms, axis=-1) * dvol

def get_dthomell(ms, nzm, probell, biases, dvol):
    # integrand in 2-halo numerator
    return np.trapz(biases * nzm * probell, ms, axis=-1) * dvol

def get_Celldtaudtau_2h(zs, ms, ks, chis, PzkLinz1z2, dtauell, ell):
    jn = spherical_jn(ell, np.outer(ks,chis))
    zis = np.arange(len(zs))
    
    Cl2hz1z2 = lambda z1, z2: 2./np.pi * dtauell[ell,z1] * dtauell[ell,z2] * np.trapz(ks**2. * PzkLinz1z2[z1,z2] * jn[:,z1] * jn[:,z2], ks, axis=-1)
    
    Cl2h = np.asarray([np.asarray([Cl2hz1z2(z1, z2) for z2 in zis]) for z1 in zis])
    for z1 in zis:
        nans, x = nan_helper(Cl2h[z1])
        Cl2h[z1][nans]= np.interp(x(nans), x(~nans), Cl2h[z1][~nans])
#        print('2h redshift', zis[z1], z1, 'ell', ell, '<<nan>> redshift indices:', nans)

    CldTau2h = np.trapz(np.trapz(Cl2h, zs, axis=-1), zs, axis=-1)
    if ell%100==0: print(ell)
    return CldTau2h * (4.*np.pi) / (2.*ell+1.)

def get_Celldtaudthom_2h(zs, ms, ks, chis, PzkLinz1z2, dthomell, dtauell, ell):
    jn = spherical_jn(ell, np.outer(ks,chis))
    zis = np.arange(len(zs))

    Cl2hz1z2 = lambda z1, z2: 2./np.pi * dthomell[ell,z1] * dtauell[ell,z2] * np.trapz(ks**2. * PzkLinz1z2[z1,z2] * jn[:,z1] * jn[:,z2], ks, axis=-1)

    Cl2h = np.asarray([np.asarray([Cl2hz1z2(z1, z2) for z2 in zis]) for z1 in zis])
    for z1 in zis:
        nans, x = nan_helper(Cl2h[z1])
        Cl2h[z1][nans]= np.interp(x(nans), x(~nans), Cl2h[z1][~nans])
#        print('2h redshift', zis[z1], z1, 'ell', ell, '<<nan>> redshift indices:', nans)

    CldTau2h = np.trapz(np.trapz(Cl2h, zs, axis=-1), zs, axis=-1)
    if ell%100==0: print(ell)
    return CldTau2h * ((4.*np.pi) / (2.*ell+1.))**0.5

def get_Celldthomdthom_2h(zs, ms, ks, chis, PzkLinz1z2, dthomell, ell):
    jn = spherical_jn(ell, np.outer(ks,chis))
    zis = np.arange(len(zs))

    Cl2hz1z2 = lambda z1, z2: 2./np.pi * dthomell[ell,z1] * dthomell[ell,z2] * np.trapz(ks**2. * PzkLinz1z2[z1,z2] * jn[:,z1] * jn[:,z2], ks, axis=-1)
    
    Cl2h = np.asarray([np.asarray([Cl2hz1z2(z1, z2) for z2 in zis]) for z1 in zis])
    for z1 in zis:
        nans, x = nan_helper(Cl2h[z1])
        Cl2h[z1][nans]= np.interp(x(nans), x(~nans), Cl2h[z1][~nans])
#        print('2h redshift', zis[z1], z1, 'ell', ell, '<<nan>> redshift indices:', nans)

    CldTau2h = np.trapz(np.trapz(Cl2h, zs, axis=-1), zs, axis=-1)
    if ell%100==0: print(ell)
    return CldTau2h

def get_scrCLs(l0Max, l1Max, l2Max, CMB, DPCl):
    l0List   = np.arange(   l0Max)
    ell1_CMB = np.arange(2, l1Max)
    ell2_scr = np.arange(2, l2Max)

    TTCl, EECl, BBCl, TECl = CMB[:,0], CMB[:,1], CMB[:,2], CMB[:,3]

    every_pair = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    allcomb = len(every_pair)
    nums    = np.array(np.linspace(0, allcomb, 20), dtype=int).tolist()

    scrTT, scrEE, scrBB, scrTE = np.zeros(l0Max), np.zeros(l0Max), np.zeros(l0Max), np.zeros(l0Max)
    wig000       = np.zeros(l0Max)
    wig220       = np.zeros(l0Max)
    for ind, (l1,l2) in enumerate(every_pair):

        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        norm = (2.*l1+1.)*(2.*l2+1.)/(4.*np.pi)

        w000   = wigner.wigner_3jj(l1, l2, 0, 0)
        am, bm = max(2, int(w000[0])), min(int(w000[1]), l0Max-1)
        l000   = np.arange(am, bm+1)
        aw, bw  = int(am - w000[0]), int(bm - w000[0])

        wig000[:] = 0.
        wig000[l000] = w000[2][aw:bw+1]

        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw  = int(cm - w220[0]), int(dm - w220[0])

        wig220[:] = 0.
        wig220[l220] = w220[2][cw:dw+1]

        scrTT += norm * DPCl[l2] * TTCl[l1] * wig000**2.
        scrTE += norm * DPCl[l2] * TECl[l1] * wig000 * wig220

        mix    = norm * DPCl[l2] * EECl[l1] * wig220**2.
        Jell   = l0List+l1+l2
        delte  = 0.5*(1. + (-1.)**Jell)
        delto  = 0.5*(1. - (-1.)**Jell)

        scrEE += mix * delte
        scrBB += mix * delto
    return l0List, scrTT, scrEE, scrBB, scrTE


def get_Tmonopole_screeningPS(l0Max, l2Max, CMB, DPCl):
    l0List   = np.arange(   l0Max)
    ell2_scr = np.arange(2, l2Max)
    TTC0 = CMB[0,0]

    scrTT = np.zeros(l0Max)
    wig000 = np.zeros(l0Max)
    for l2 in ell2_scr:
        norm = (2.*l2+1.)*(4.*np.pi)**-1.
        w000 = wigner.wigner_3jj(0, l2, 0,  0)
        am, bm  = max(2, int(w000[0])), min(int(w000[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - w000[0]), int(bm - w000[0])
        wig000[:]   = 0.
        wig000[l0list] = w000[2][aw:bw+1]
        scrTT += norm * DPCl[l2] * TTC0 * wig000**2.
    return l0List, scrTT


def noise(ells, experiment):
    # Instrumental noise: takes parameters Beam FWHM and Experiment sensitivity in T
    ''' Output format: (spectrum type, ells, channels)'''

    beamFWHM = experiment['FWHMrad']**2.
    deltaT   = experiment['SensitivityμK']**2.
    lknee    = experiment['Knee ell']
    aknee    = experiment['Exponent']

    NellTT = np.zeros((len(ells), len(beamFWHM)))
    for frq in range(len(NellTT[0])):
        Beam = ells[2:]*(ells[2:]+1.) * beamFWHM[frq]/np.log(2)/8.
        WNoise = deltaT[frq] * ( 1. + (ells[2:]/lknee[frq])**aknee[frq] )
        NellTT[2:, frq] = WNoise * np.exp(Beam)
    return np.asarray([NellTT, 2.**0.5 * NellTT, 2.**0.5 * NellTT])


def get_ILC_weights(ellMax, Nell, recCMB, screening, freqMAT):
    ells = np.arange(ellMax)
    ee = np.ones(len(freqMAT[0]))
    weights = np.ones(np.shape(Nell))
    onesMAT = np.outer(ee, ee)

    leftover = np.zeros(np.shape(Nell[:,:,0]))
    for spec in range(len(Nell)):
        for ell in ells[2:ellMax]:
            Nellω2   = np.diag(np.diag(freqMAT) * Nell[spec, ell])
            CellBBω2 = freqMAT * recCMB[ell, spec]
            Cellττ  = onesMAT * screening[spec, ell]
            Cellinv = scp.linalg.inv(Cellττ + CellBBω2 + Nellω2)
            weights = (Cellinv@ee)/(ee@Cellinv@ee)
            leftover[spec, ell] = weights@(Nellω2 + CellBBω2)@weights
    return leftover

def return_ILC_weights(ellMax, baseline, screening, recCMB, experiment):
    ells = np.arange(ellMax)
    Nell  = noise(ells, experiment)
    freqs = experiment['freqseV']
    freqMAT = np.outer(freqs/baseline, freqs/baseline)

    ee = np.ones(len(freqMAT[0]))
    weights = np.ones(np.shape(Nell))
    onesMAT = np.outer(ee, ee)

    weights = np.zeros(np.shape(Nell))
    for spec in range(len(Nell)):
        for ell in ells[2:ellMax]:
            Nellω2   = np.diag(np.diag(freqMAT) * Nell[spec, ell])
            CellBBω2 = freqMAT * recCMB[ell, spec]
            Cellττ  = onesMAT * screening[spec, ell]
            Cellinv = scp.linalg.inv(Cellττ + CellBBω2 + Nellω2)
            weights[spec, ell] = (Cellinv@ee)/(ee@Cellinv@ee)
    return weights

def get_ILC_noise(ellMax, baseline, screening, recCMB, experiment):
    ells = np.arange(ellMax)
    Nell  = noise(ells, experiment)
    freqs = experiment['freqseV']
    freqMAT = np.outer(freqs/baseline, freqs/baseline)

    leftover = get_ILC_weights(ellMax, Nell, recCMB, screening, freqMAT)
    return leftover


def get_ILC_BB_weights(ellMax, Nell, recCMB, screening, ee):
    ells = np.arange(ellMax)
    mat = np.outer(ee, ee)

    leftover = np.zeros((len(Nell), ellMax))
    for spec in range(len(Nell)):
        for ell in ells[2:ellMax]:
            Nellω2   = np.diag(Nell[spec, ell])
            CellBBω2 = mat * (recCMB[ell, spec] + screening[spec, ell])
            Cellinv  = scp.linalg.inv(CellBBω2 + Nellω2)
            weights  = (Cellinv@ee)/(ee@Cellinv@ee)
            leftover[spec, ell] = weights@(Nellω2 + CellBBω2)@weights
    return leftover

def return_ILC_BB_weights(ellMax, screening, recCMB, experiment):
    ells = np.arange(ellMax)
    Nell  = noise(ells, experiment)
    freqs = experiment['freqseV']
    ee  = np.ones(len(freqs))
    mat = np.outer(ee, ee)

    weights = np.zeros(np.shape(Nell))
    for spec in range(len(Nell)):
        for ell in ells[2:ellMax]:
            Nellω2   = np.diag(Nell[spec, ell])
            CellBBω2 = mat * (recCMB[ell, spec] + screening[spec, ell])
            Cellinv  = scp.linalg.inv(CellBBω2 + Nellω2)
            weights[spec, ell] = (Cellinv@ee)/(ee@Cellinv@ee)
    return weights

def get_ILC_BB_noise(ellMax, screening, recCMB, experiment):
    ells = np.arange(ellMax)
    Nell  = noise(ells, experiment)
    freqs = experiment['freqseV']
    ee    = np.ones(len(freqs))
    leftover = get_ILC_BB_weights(ellMax, Nell, recCMB, screening, ee)
    return leftover


def taureco_NEdscBdsc(l0Max, l1Max, l2Max, ClEErec, ClBBrec, NlEE, NlBB):
    l0List = np.arange(   l0Max)
    l1List = np.arange(2, l1Max)
    l2List = np.arange(2, l2Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))

    sumell = np.zeros(l0Max)
    for l1,l2 in all_possible_pairs:
        wig3j   = wigner.wigner_3jj(l1, l2, -2,  2)
        wig3jre = wigner.wigner_3jj(l1, l2,  2, -2)

        am, bm  = int(wig3j[0]), min(int(wig3j[1])+1, l0Max-1)
        lwig    = np.arange(am, bm)

        pad       = np.zeros(l0Max)
        pad[lwig] = wig3j[2][lwig-am] - wig3jre[2][lwig-am]

        norm    = (2.*l1+1.)*(2.*l2+1.)*(2.*l0List+1.)/(4.*np.pi)
        gaEB    = norm * np.abs(-0.5j * ClEErec[l1] * pad)**2.
        #denom   = (ClEErec[l1] + NlEE[l1])*(ClBBrec[l2] + NlBB[l2])
        denom   = NlEE[l1] * NlBB[l2]
        sumell += gaEB / denom
    return (2.*l0List+1.) / sumell

def bispectrum_Tdsc_Tsc_Tsc(l0Max, l1Max, l2Max, T0, ClTT, Cltautaurei, NICLdscdsc, NICLscsc):
    l1List = np.arange(2, l1Max)
    l2List = np.arange(2, l2Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))
    sumlist = np.empty(len(all_possible_pairs))

    for ind, (l1,l2) in enumerate(all_possible_pairs):
        wig3j = wigner.wigner_3jj(l2, l1, 0,  0)

        am, bm  = max(2, int(wig3j[0])), min(int(wig3j[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - wig3j[0]), int(bm - wig3j[0])
        wig000  = wig3j[2][aw:bw+1]

        norm   = (wig000 * T0)**2. * (2.*l0list+1.)
        numer  = norm * ((ClTT[l0list] + ClTT[l1]) * Cltautaurei[l2])**2.
        denom  = NICLdscdsc[l2] * NICLscsc[l0list] * NICLscsc[l1]

        sumlist[ind] = np.sum(numer / denom)
    return np.sum(sumlist)

def bispectrum_Tdsc_Esc_Bsc(l0Max, l1Max, l2Max, T0, ClEE, Cltautaurei, NICLdscdsc, NICLEEscsc, NICLBBscsc):
    l1List = np.arange(2, l1Max)
    l2List = np.arange(2, l2Max)

    all_possible_pairs = np.asarray(list(itertools.product(l1List, l2List)))
    sumlist = np.empty(len(all_possible_pairs))

    for ind, (l1,l2) in enumerate(all_possible_pairs):
        wig3j = wigner.wigner_3jj(l2, l1, 0,  2)

        am, bm  = max(2, int(wig3j[0])), min(int(wig3j[1]), l0Max-1)
        l0list  = np.arange(am, bm+1)
        aw, bw  = int(am - wig3j[0]), int(bm - wig3j[0])
        wig220  = wig3j[2][aw:bw+1]

        Jell   = l0list+l1+l2
        delto  = 0.5*(1. - (-1.)**Jell)

        norm   = delto * (wig220 * T0)**2. * (2.*l0list+1.)
        numer  = norm * ((ClEE[l0list] + ClEE[l1]) * Cltautaurei[l2])**2.
        denom  = NICLdscdsc[l2] * NICLBBscsc[l0list] * NICLEEscsc[l1]

        sumlist[ind] = np.sum(numer / denom)
    return np.sum(sumlist)


def w000(Ell, ell0, ell1, ell2):
    # very fast wigner 3j with m1 = m2 = m3 = 0
    g = Ell/2.
    w = np.exp(0.5*(lgamma(2.*g-2.*ell0+1.)+lgamma(2.*g-2.*ell1+1.)+lgamma(2.*g-2.*ell2+1.)-lgamma(2.*g+2.))\
                          +lgamma(g+1.)-lgamma(g-ell0+1.)-lgamma(g-ell1+1.)-lgamma(g-ell2+1.))
    return w * (-1.)**g


########## Covariance Matrices + Fischer forecasting ###########

def sigma_screening(epsilon4, ellmin, ellmax, screening, leftover):
    ClTTNl = epsilon4 * screening[:, 0] + leftover[:, 0]
    ClEENl = epsilon4 * screening[:, 1] + leftover[:, 1]
    ClBBNl = epsilon4 * screening[:, 2] + leftover[:, 2]

    dClTTde4 = screening[:, 0]
    dClEEde4 = screening[:, 1]
    dClBBde4 = screening[:, 2]

    TrF = np.empty(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[ClTTNl[el], 0.        , 0.        ],\
                           [0.        , ClEENl[el], 0.        ],\
                           [0.        , 0.        , ClBBNl[el]]])
        CCovInv = np.linalg.inv(CCov)
        dCovde4 = np.asarray([[dClTTde4[el], 0.          , 0.          ],\
                              [0.          , dClEEde4[el], 0.          ],\
                              [0.          , 0.          , dClBBde4[el]]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde4@CCovInv@dCovde4)
    return 0.21 * (np.sum(TrF[ellmin:])**(-0.25))**0.5

def sigma_screeningVtemplate(TCMB, epsilon2, ellmin, ellmax, screeningCosmo, leftover, cltautauSurvey):
    ClTTNl = epsilon2**2. * screeningCosmo[:, 0] + leftover[:, 0]
    Clττ = cltautauSurvey
    ClTτscr = epsilon2 * TCMB * cltautauSurvey
    dClTTde2 = 2.*epsilon2 * screeningCosmo[:, 0]
    dClTτscrde2 = TCMB * cltautauSurvey

    TrF = np.empty(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[Clττ[el]   , ClTτscr[el]],\
                           [ClTτscr[el], ClTTNl[el] ]])
        CCovInv = np.linalg.inv(CCov)
        dCovde2 = np.asarray([[0.             , dClTτscrde2[el]],\
                              [dClTτscrde2[el], dClTTde2[el]   ]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde2@CCovInv@dCovde2)
    return 0.35 * np.sum(TrF[ellmin:])**(-0.25)

def battagliaAGN(m200, zs):
    # power law fits:
    rho0  = 4000. * (m200/1e14)**0.29    * (1.+zs)**(-0.66)
    alpha = 0.88  * (m200/1e14)**(-0.03) * (1.+zs)**0.19
    beta  = 3.83  * (m200/1e14)**0.04    * (1.+zs)**(-0.025)
        
    gamma = -0.2
    xc    = 0.5
    return rho0, alpha, beta, gamma, xc

def battagliaSH(m200, zs):
    # power law fits:
    rho0  = 1.9e4 * (m200/1e14)**0.09     * (1.+zs)**(-0.95)
    alpha = 0.7   * (m200/1e14)**(-0.017) * (1.+zs)**0.27
    beta  = 4.43  * (m200/1e14)**0.005    * (1.+zs)**0.037

    gamma = -0.2
    xc    = 0.5
    return rho0, alpha, beta, gamma, xc

def bestfitACT():
    rho0  = np.exp(2.6*np.log(10))
    alpha = 1.
    beta  = 2.6
    gamma = -0.2
    xc    = 0.6
    return rho0, alpha, beta, gamma, xc
