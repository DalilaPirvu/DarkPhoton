import os,sys
sys.path.append('../')
import hmvec as hm # Git clone and pip install as in readme from github.com/msyriac/hmvec
import numpy as np
#import constants # this should be in the same parent folder; contains constants used in calculations
from scipy.special import eval_legendre
from scipy.special import spherical_jn
from sympy.physics.wigner import wigner_3j as wigner3j

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
aa = lambda z: 1./(1.+z)
omegaz = lambda z: omega0/aa(z)


############### COMPUTE ANGULAR POWER SPECTRA ###########################

# Compute crossing radius of each halo
def dark_photon_conv(zs, ms):
    NZ, NM = len(zs), len(ms)
    rcross = np.zeros((NZ,NM))       # Array for radius where plasma mass^2 = A' mass^2
    chicross = np.zeros((NZ,NM))     # Array for radius where plasma mass^2 = A' mass^2
    prob = np.zeros((NZ,NM))         # Array for optical depth per halo due to A' conversion.
    cs = hcos.concentration()

    for zi, zz in enumerate(zs):
        rvir = hcos.rvir(ms,zz)
        rss = rvir/cs[zi]
        nfw_rhoscsales = hm.rhoscale_nfw(ms, rvir, cs[zi])
        nfw_profiles = np.asarray([hm.rho_nfw(rs, nfw_rhoscsales[mi], rss[mi]) for mi in range(NM)])

        # Find the index of the radius array where plasmon mass^2 = dark photon mass^2
        idx = np.argmin(np.abs(conv*nfw_profiles - MA**2), axis=1)

        # Fill the array with crossing radii, one for each halo
        rcross[zi] = rs[idx]
        chicross[zi] = rcross[zi]*(1+zz)
        # d m_gamma^2 / dr evaluated where plasma mass^2 = A' mass^2
        rfr = rcross[zi]/rss
        dmdr = conv*(nfw_rhoscsales/rss)*(1+3.*rfr)/(rfr)**2./(1+rfr)**3.

        # Conversion probability per halo, ensure (rcross < r_vir) with Heaviside function
        prob[zi] = mpcEVinv*np.pi*MA**4.*np.heaviside(rvir-rcross[zi], 0.5)/omegaz(zz)/dmdr
    return rcross, chicross, prob

# angular function u(theta); the prefactor of the Y_(ell,0) np.sqrt((2*ell+1)/4/np.pi)
# cancels out with the same factor in the definition of the CL's later on
def get_angular_u(ells, redshifts, conv_radii):
    rchi = hcos.comoving_radial_distance(redshifts)*aa(redshifts)      # proper distance from observer to redshift bin in Mpc
    rchi2D = np.outer(rchi, np.ones(conv_radii.shape[-1]))             # dimensions:(nZs,nMasses)

    angs = np.linspace(0, (1.-1e-16)*np.arcsin(conv_radii/rchi2D), 2000)
    sin_angs = np.sin(angs)
    cos_angs = np.cos(angs)
    sqrt_dist = np.sqrt(1.-(rchi2D*sin_angs/conv_radii)**2.)
    norm = 2.*np.pi*np.trapz(sin_angs*sqrt_dist, angs, axis=0)
    uu = sqrt_dist / norm
    uL0 = lambda ell: 2.*np.pi*np.trapz(sin_angs * uu * eval_legendre(ell, cos_angs), angs, axis=0)
    return np.asarray([uL0(ell) for ell in ells])

def spectrum_conv(ells, zs, ks, ms, chis, Prob, Uang):
    chi = hcos.comoving_radial_distance(zs)
    nzm = hcos.get_nzm()
    PzkLin = hcos._get_matter_power(zs,ks,nonlinear=False)
    biases = hcos.get_bh()
    Hz = hcos.h_of_z(zs)

    # Fractional area of halo on the sky multiplied by area element on redshift bin / H(z)
    harea = np.pi * chis**2. / np.outer(Hz, np.ones(chis.shape[-1]))

    # Average optical depth per redshift bin
    avtau = np.trapz(nzm * Prob * harea, ms, axis=-1)

    # The 1-halo term
    CL1h = np.trapz(nzm*(Prob*harea*np.abs(Uang))**2., ms, axis=-1) / avtau**2.
    CL1h[np.isnan(CL1h)] = 0.

    # The 2-halo term
    num2h = np.trapz(nzm*Prob*harea*biases*Uang, ms, axis=-1)

    CLm = np.asarray([[np.sqrt(PzkLin[zi1,:]*PzkLin[zi2,:]) * ks**2. for zi2 in range(nZs)] for zi1 in range(nZs)])
    CLhh = lambda zi1, zi2, ell: 2./np.pi * np.trapz(CLm[zi1, zi2] * spherical_jn(int(ell), ks*np.abs(chi[zi1]-chi[zi2]))**2., ks, axis=-1)
    CLhhlist = np.asarray([[[CLhh(zi1, zi2, ell) for zi2 in range(nZs)] for zi1 in range(nZs)] for ell in ells])

    CL2h = lambda zi1, zi2, ell: CLhhlist[ell, zi1, zi2] * num2h[ell, zi1]/avtau[zi1] * num2h[ell, zi2]/avtau[zi2]
    CL2hlist = np.asarray([[[CL2h(zi1, zi2, ell) for zi2 in range(nZs)] for zi1 in range(nZs)] for ell in range(len(ells))])
    CL2hlist[np.isnan(CL2hlist)] = 0.

    CLdTau1h = np.trapz(CL1h, zs, axis=-1)
    CLdTau2h = np.trapz(np.trapz(CL2hlist, zs, axis=-1), zs, axis=-1)
    return avtau, CL1h, CLhhlist, CL2hlist, CLdTau1h, CLdTau2h, CLdTau1h+CLdTau2h


############### Compute CMB angular power spectra

# vectors are formatted as [TT,EE,BB,TE]
def get_scrCLs(ellList, screeningCL, CMBCL, ellmaxout):
    NElls = ellList[-1]
    CLdeltaTau, CLscrTT, CLscrEE, CLscrBB, CLscrTE = (np.zeros(NElls) for i in range(5))
    CLdeltaTau[ellList] = screeningCL

    lists0 = np.asarray(list(itertools.product(range(ellmaxout), range(ellmaxout))))
    listsmax = np.sum(lists0, axis=-1)
    listsmin = np.abs(lists0[:,0]-lists0[:,1])

    totsteps = len(lists0)
    for ind, (l_min, l_max) in enumerate(zip(listsmin, listsmax)):
        if ind%10000==0: print('done', ind, 'out of ', totsteps)

        ell1, ell2 = lists0[ind]
        pref = (2.*ell1+1.)*(2.*ell2+1.)/4./np.pi

        ell0list = ellList[(ellList>=l_min)&(ellList<=l_max)]
        for ell0 in ell0list:
            Ell = ell0+ell1+ell2

            w220 = float(wigner3j(ell0,ell1,ell2,-2,2,0))
            mix = CLdeltaTau[ell2] * CMBCL[ell1,1] * pref * w220**2.

            if Ell%2==0:
                w000 = float(wigner3j(ell0,ell1,ell2,0,0,0))
                CLscrTT[ell0] += CLdeltaTau[ell2] * CMBCL[ell1,0] * pref * w000**2.
                CLscrTE[ell0] += CLdeltaTau[ell2] * CMBCL[ell1,3] * pref * w000 * w220
                CLscrEE[ell0] += mix
            else:
                CLscrBB[ell0] += mix
    return np.asarray([CLscrTT[::dell], CLscrEE[::dell], CLscrBB[::dell], CLscrTE[::dell]])



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

    test1_cells_CMBDP = get_TEST1_CMB_dark_photons()
    np.save(dirdata(MA, omega0, nZs, nMasses, nElls, dell)+'CMBDP_test1', test1_cells_CMBDP)

    test2_cells_CMBDP = get_TEST2_CMB_dark_photons()
    np.save(dirdata(MA, omega0, nZs, nMasses, nElls, dell)+'CMBDP_test2', test2_cells_CMBDP)
