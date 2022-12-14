import os,sys
sys.path.append('../')
import hmvec as hm # Git clone and pip install as in readme from github.com/msyriac/hmvec
import numpy as np
import itertools
#import constants # this should be in the same parent folder; contains constants used in calculations
from scipy.special import eval_legendre
from scipy.special import spherical_jn
import pywigxjpf as wig
from math import lgamma as lgamma 

wig.wig_table_init(2*10000,3) # should be >= ellMax
wig.wig_temp_init(2*10000)

############# Define constants ################
# From hmvec:
# proper radius r is always in Mpc
# comoving momentum k is always in Mpc-1
# All masses m are in Msolar
# rho densities are in Msolar/Mpc^3
# No h units anywhere

cmMpc = 3.24e-25               # Mpc/cm            # how many Mpc in a cm
eVinvCm = 1.24e-4              # cm/eV^-1          # how many cm in a eV^-1
mpcEVinv = 1./(cmMpc*eVinvCm)  # eV^-1/Mpc         # how many eV^-1 in a Mpc

msun = 1.9891e30     # kg                # Sun mass
mprot = 1.6e-27      # kg                # Proton mass
m2eV = 1.4e-21       # eV^2              # conversion factor for plasma mass (eq. (2) in Caputo et al; PRL)
ombh2 = 0.02225                          # Physical baryon density parameter Ωb h2
omch2 = 0.1198                           # Physical dark matter density parameter Ωc h2
conv = m2eV*(ombh2/omch2)*(msun/mprot)*(cmMpc)**3

############### COMPUTE ANGULAR POWER SPECTRA ###########################

# Compute crossing radius of each halo
def dark_photon_conv_prob(zs, ms, rs, rvir, cs, Hz, mDP, omega0):
    NZ, NM = len(zs), len(ms)
    rcross = np.zeros((NZ,NM))       # Array for radius where plasma mass^2 = A' mass^2
    chicross = np.zeros((NZ,NM))     # Array for radius where plasma mass^2 = A' mass^2
    prob = np.zeros((NZ,NM))         # Array for optical depth per halo due to A' conversion.

    for zi, zz in enumerate(zs):
        rss = rvir[zi]/cs[zi]
        nfw_rhoscsales = hm.rhoscale_nfw(ms, rvir[zi], cs[zi])
        nfw_profiles = np.asarray([hm.rho_nfw(rs, nfw_rhoscsales[mi], rss[mi]) for mi in range(NM)])

        # Find the index of the radius array where plasmon mass^2 = dark photon mass^2
        idx = np.argmin(np.abs(conv*nfw_profiles - mDP**2), axis=1)

        # Fill the array with crossing radii, one for each halo
        rcross[zi] = rs[idx]
        chicross[zi] = rcross[zi]*(1+zz)
        # d m_gamma^2 / dr evaluated where plasma mass^2 = A' mass^2
        rfr = rcross[zi]/rss
        dmdr = conv*(nfw_rhoscsales/rss)*(1+3.*rfr)/(rfr)**2./(1+rfr)**3.

        # Conversion probability per halo, ensure (rcross < r_vir) with Heaviside function
        omegaz = omega0*(1+zz)
        prob[zi] = mpcEVinv*np.pi*mDP**4.*np.heaviside(rvir[zi]-rcross[zi], 0.5)/omegaz/dmdr

    # Fractional area of halo on the sky multiplied by area element on redshift bin / H(z)
    harea = np.pi * chicross**2. / np.outer(Hz, np.ones(chicross.shape[-1]))
    return rcross, prob, harea

# angular function u(theta); the prefactor of the Y_(ell,0) np.sqrt((2*ell+1)/4/np.pi)
# cancels out with the same factor in the definition of the CL's later on
def get_angular_u_ell(zs, rchi, cross_rad, ell):
    # dimensions:(nZs,nMasses)
    rchi2D = np.outer(rchi, np.ones(cross_rad.shape[-1]))
    angs = np.linspace(0., (1.-1e-16)*np.arcsin(cross_rad/rchi2D), 2000)
    sin_angs, cos_angs = np.sin(angs), np.cos(angs)
    sqrt_dist = np.sqrt(1.-(rchi2D*sin_angs/cross_rad)**2.)
    norm = 1./np.trapz(sin_angs*sqrt_dist, angs, axis=0)
    uL0 = norm * np.trapz(sin_angs*sqrt_dist*eval_legendre(ell, cos_angs), angs, axis=0)
    return uL0

def get_scr_powspec_1h(zs, ks, ms, nzm, biases, harea, Prob, Uang):
    # Average optical depth per redshift bin
    avtau = np.trapz(nzm * Prob * harea, ms, axis=-1)

    # 2-halo numerator
    num2h = np.trapz(nzm * Prob * harea * biases * Uang, ms, axis=-1)

    # The 1-halo term
    CL1h = np.trapz(nzm * (Prob * harea * np.abs(Uang))**2., ms, axis=-1) / avtau**2.
    CL1h[np.isnan(CL1h)] = 0.
    CLdTau1h = np.trapz(CL1h, zs, axis=-1)
    return avtau, num2h, CLdTau1h

def get_scr_powspec_2h(zs, ks, chis, avtau, PzkLin, num2h, ell):
    CLhhz1z2 = lambda ell, z1, z2: 2./np.pi*np.trapz(Clmat[z1,z2] * spherical_jn(ell, ks*np.abs(chis[z1]-chis[z2]))**2., ks, axis=-1)
    CL2hz1z2 = lambda ell, z1, z2: CLhh[z1,z2] * num2h[ell,z1]/avtau[z1] * num2h[ell,z2]/avtau[z2]

    nZs = len(zs)
    Clmat = np.asarray([[ks**2.*np.sqrt(PzkLin[z1,:]*PzkLin[z2,:]) for z2 in range(nZs)] for z1 in range(nZs)])
    CLhh = np.asarray([[CLhhz1z2(ell, z1, z2) for z2 in range(nZs)] for z1 in range(nZs)])
    CL2h = np.asarray([[CL2hz1z2(ell, z1, z2) for z2 in range(nZs)] for z1 in range(nZs)])
    CL2h[np.isnan(CL2h)] = 0.
    CLdTau2h = np.trapz(np.trapz(CL2h, zs, axis=-1), zs, axis=-1)
    return CLhh, CLdTau2h


def wigner_symbol000(Ell, ell0, ell1, ell2):
    g = Ell/2.
    w = (-1.)**(g)*np.exp((lgamma(2.*g-2.*ell0+1.)+lgamma(2.*g-2.*ell1+1.)+lgamma(2.*g-2.*ell2+1.)\
        -lgamma(2.*g+2.))/2.+lgamma(g+1.)-lgamma(g-ell0+1.)-lgamma(g-ell1+1.)-lgamma(g-ell2+1.))
    return w

def parallelly_get_scrCLs(DPCl, CMBCl, ell2max, ell0):   
    CLscrTT, CLscrEE, CLscrBB, CLscrTE = 0.,0.,0.,0.
    ell2_scr = np.arange(2, ell2max)
    ell1_CMB = np.arange(2, 2401)
    
    all_possible_pairs = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    listsmax = np.sum(all_possible_pairs, axis=-1)
    listsmin = np.abs(all_possible_pairs[:,0] - all_possible_pairs[:,1])

    for ind, (ell1,ell2) in enumerate(all_possible_pairs):
        if (listsmin[ind] <= ell0) and (ell0 <= listsmax[ind]):

            pref = (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi
            Ell = ell0+ell1+ell2

            w220 = wig.wig3jj(2*ell0, 2*ell1, 2*ell2, -4, 4, 0)
            mix = DPCl[ell2] * CMBCl[ell1,1] * pref * w220**2.

            if Ell%2==0:
                w000 = wigner_symbol000(Ell, ell0, ell1, ell2)
                CLscrTT += DPCl[ell2] * CMBCl[ell1,0] * pref * w000**2.
                CLscrTE += DPCl[ell2] * CMBCl[ell1,3] * pref * w000 * w220
                CLscrEE += mix
            else:
                CLscrBB += mix
    return np.asarray([CLscrTT, CLscrEE, CLscrBB, CLscrTE])





####################################################################
####################################################################
####                      Older versions                      ######
####################################################################
####################################################################

# Check code: compute CMBDP with constant conv Cells
testt = False

if testt:

    eh = lambda Ell: 0.5*(1.+(-1.)**Ell) # which is 1 for Ell even and 0 for Ell odd
    oh = lambda Ell: 0.5*(1.-(-1.)**Ell) # which is 0 for Ell even and 1 for Ell odd

    W000sq = lambda ell0, ell1, ell2: (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi * wigner3j(ell0,ell1,ell2,0,0,0)**2.
    ehW220sq = lambda ell0,ell1,ell2: (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi * eh(ell0+ell1+ell2) * wigner3j(ell0,ell1,ell2,-2,2,0)**2.
    ohW220sq = lambda ell0,ell1,ell2: (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi * oh(ell0+ell1+ell2) * wigner3j(ell0,ell1,ell2,-2,2,0)**2.
    W000W220 = lambda ell0,ell1,ell2: (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi * wigner3j(ell0,ell1,ell2,0,0,0) * wigner3j(ell0,ell1,ell2,-2,2,0)

    CltautauClTTelem = lambda ell0, ell1, ell2: cellDeltaTau[ells.tolist().index(ell2)] * unlensedCL[ell1,0] * W000sq(ell0,ell1,ell2)
    CltautauClEEelem = lambda ell0, ell1, ell2: cellDeltaTau[ells.tolist().index(ell2)] * unlensedCL[ell1,1] * ehW220sq(ell0,ell1,ell2)
    CltautauClBBelem = lambda ell0, ell1, ell2: cellDeltaTau[ells.tolist().index(ell2)] * unlensedCL[ell1,1] * ohW220sq(ell0,ell1,ell2)
    CltautauClTEelem = lambda ell0, ell1, ell2: cellDeltaTau[ells.tolist().index(ell2)] * unlensedCL[ell1,3] * W000W220(ell0,ell1,ell2)

    def get_TEST1_CMB_dark_photons():
        CltautauClTT, CltautauClEE, CltautauClBB, CltautauClTE = np.zeros(nElls), np.zeros(nElls), np.zeros(nElls), np.zeros(nElls)
        for ell0 in ells:
            CltautauClTT[ell0] = np.sum([np.sum([CltautauClTTelem(ell0,ell1,ell2) if (ell1+ell2>=ell0) and (np.abs(ell1-ell2)<=ell0) and ((ell0+ell1+ell2)%2==0) else 0. for ell2 in ells[1:]], axis=0) for ell1 in ells[1:]], axis=0)
            CltautauClEE[ell0] = np.sum([np.sum([CltautauClEEelem(ell0,ell1,ell2) if (ell1+ell2>=ell0) and (np.abs(ell1-ell2)<=ell0) else 0. for ell2 in ells[1:]], axis=0) for ell1 in ells[1:]], axis=0)
            CltautauClBB[ell0] = np.sum([np.sum([CltautauClBBelem(ell0,ell1,ell2) if (ell1+ell2>=ell0) and (np.abs(ell1-ell2)<=ell0) else 0. for ell2 in ells[1:]], axis=0) for ell1 in ells[1:]], axis=0)
            CltautauClTE[ell0] = np.sum([np.sum([CltautauClTEelem(ell0,ell1,ell2) if (ell1+ell2>=ell0) and (np.abs(ell1-ell2)<=ell0) and ((ell0+ell1+ell2)%2==0) else 0. for ell2 in ells[1:]], axis=0) for ell1 in ells[1:]], axis=0)
        return np.asarray([CltautauClTT[::dell], CltautauClEE[::dell], CltautauClBB[::dell], CltautauClTE[::dell]])

    def get_TEST2_CMB_dark_photons():
        CltautauClTT, CltautauClEE, CltautauClBB, CltautauClTE = np.zeros(nElls), np.zeros(nElls), np.zeros(nElls), np.zeros(nElls)
        # duration scales as nElls**3
        # but it only scans for selection rule once for all CL
        lists0 = list(itertools.product(ells[ells>0.], ells[ells>0.]))
        for ell0 in ells:
            list_nonzero = [elem for elem in lists0 if sum(elem)>=ell0 and np.abs(elem[0]-elem[1])<=ell0]
            list_even = [elem for elem in list_nonzero if (sum(elem)+ell0)%2==0]
            print('ell0 =', ell0, len(list_nonzero), len(list_even))

            CltautauClTT[ell0] = sum(CltautauClTTelem(ell0,kk[0],kk[1]) for kk in list_even)
            CltautauClEE[ell0] = sum(CltautauClEEelem(ell0,kk[0],kk[1]) for kk in list_nonzero)
            CltautauClBB[ell0] = sum(CltautauClBBelem(ell0,kk[0],kk[1]) for kk in list_nonzero)
            CltautauClTE[ell0] = sum(CltautauClTEelem(ell0,kk[0],kk[1]) for kk in list_even)
        return np.asarray([CltautauClTT[::dell], CltautauClEE[::dell], CltautauClBB[::dell], CltautauClTE[::dell]])

    def get_TEST3_CMB_dark_photons(ellList, screeningCL, CMBCL, ellmaxout):
        NElls = ellList[-1]+1
        dell = ellList[-1]-ellList[-2]
        CLdeltaTau, CLscrTT, CLscrEE, CLscrBB, CLscrTE = (np.zeros(NElls) for i in range(5))
        CLdeltaTau[ellList] = screeningCL[:NElls]

        lists0 = np.asarray(list(itertools.product(range(ellmaxout), range(min(NElls-1, ellmaxout)))))
        listsmax = np.sum(lists0, axis=-1)
        listsmin = np.abs(lists0[:,0]-lists0[:,1])

        totsteps = len(lists0)
        for ind, (l_min, l_max) in enumerate(zip(listsmin, listsmax)):

            ell1, ell2 = lists0[ind]
            pref = (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi

            ell0list = ellList[(ellList>=l_min)&(ellList<=l_max)]
            if ind%1000000==0: print('done', ind, 'out of ', totsteps)

            for ell0 in ell0list:
                Ell = ell0+ell1+ell2

                w220 = wig.wig3jj(2*ell0, 2*ell1, 2*ell2, -4, 4, 0)
                mix = CLdeltaTau[ell2] * CMBCL[ell1,1] * pref * w220**2.

                if Ell%2==0:
                    w000 = wig.wig3jj(2*ell0, 2*ell1, 2*ell2, 0, 0, 0)
                    CLscrTT[ell0] += CLdeltaTau[ell2] * CMBCL[ell1,0] * pref * w000**2.
                    CLscrTE[ell0] += CLdeltaTau[ell2] * CMBCL[ell1,3] * pref * w000 * w220
                    CLscrEE[ell0] += mix
                else:
                    CLscrBB[ell0] += mix
        return np.asarray([CLscrTT[::dell], CLscrEE[::dell], CLscrBB[::dell], CLscrTE[::dell]])


    # vectors are formatted as [TT,EE,BB,TE]
    def get_scrCLs(ellList, DPCl, CMBCL, ellmaxout):
        NElls = ellList[-1]+1
        CLscrTT, CLscrEE, CLscrBB, CLscrTE = (np.zeros(NElls) for i in range(5))

        ell12_list = np.arange(min(NElls, ellmaxout))
        lists0 = np.asarray(list(itertools.product(ell12_list, ell12_list)))
        listsmax = np.sum(lists0, axis=-1)
        listsmin = np.abs(lists0[:,0]-lists0[:,1])

        totsteps = len(lists0)
        for ind, (l_min, l_max) in enumerate(zip(listsmin, listsmax)):
            if ind%(totsteps//10)==0: print('done', ind, 'steps out of', totsteps)

            ell1, ell2 = lists0[ind]
            pref = (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi

            ell0list = ellList[(ellList>=l_min)&(ellList<=l_max)]
            for ell0 in ell0list:
                Ell = ell0+ell1+ell2

                w220 = wig.wig3jj(2*ell0, 2*ell1, 2*ell2, -4, 4, 0)
                mix = DPCl[ell2] * CMBCL[ell1,1] * pref * w220**2.

                if Ell%2==0:
                    w000 = wigner_symbol000(Ell, ell0, ell1, ell2)
                    CLscrTT[ell0] += DPCl[ell2] * CMBCL[ell1,0] * pref * w000**2.
                    CLscrTE[ell0] += DPCl[ell2] * CMBCL[ell1,3] * pref * w000 * w220
                    CLscrEE[ell0] += mix
                else:
                    CLscrBB[ell0] += mix
        return np.asarray([CLscrTT, CLscrEE, CLscrBB, CLscrTE])


