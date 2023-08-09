import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib import gridspec
from matplotlib.gridspec import GridSpec

from mpl_toolkits.axes_grid1 import make_axes_locatable

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.1e'%x))
fmt = mticker.FuncFormatter(g)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

plt.rcParams.update({'backend' : 'Qt5Agg'})
plt.rcParams.update({'text.usetex' : True})

plt.rcParams.update({'font.size' : 11.0})
plt.rcParams.update({'axes.titlesize' : 14.0})  # Font size of title
plt.rcParams.update({'axes.titlepad'  : 10.0})
plt.rcParams.update({'axes.labelsize' : 14.0})  # Axes label sizes
plt.rcParams.update({'axes.labelpad'  : 10.0})
plt.rcParams.update({'xtick.labelsize'  : 14.0})
plt.rcParams.update({'ytick.labelsize'  : 14.0})
plt.rcParams.update({'xtick.labelsize'  : 10.0})
plt.rcParams.update({'ytick.labelsize'  : 10.0})

plt.rcParams.update({'axes.spines.left'  : True})
plt.rcParams.update({'axes.spines.right'  : True})
plt.rcParams.update({'axes.spines.top'  : True})
plt.rcParams.update({'axes.spines.bottom'  : True})
plt.rcParams.update({'savefig.format'     : 'pdf'})
plt.rcParams.update({'savefig.bbox'       : 'tight'})
plt.rcParams.update({'savefig.pad_inches' : 0.1})
plt.rcParams.update({'pdf.compression' : 6})
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#672766', '#a6228c', '#ed7625', '#1b988c', '#75bc43'])
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#b974c6', '#510c5e', '#f96a35', '#96ca4f', '#f45523'])
#plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226'])

def plot_dmdz(ms, zs, func, count=10, title='No Title'):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    for mi, mm in enumerate(ms):
        if mm in ms[:count]:
            ax[0].plot(zs, func[:,mi], linewidth=1, label=('$m=${}'.format(fmt(mm)) if mm in ms[:count//count] else None), color='k')
        elif mm in ms[-count:]:
            ax[0].plot(zs, func[:,mi], linewidth=1, label=('$m=${}'.format(fmt(mm)) if mm in ms[-count//count:] else None), color='g')
    for zi, zz in enumerate(zs):
        if zz in zs[:count]:
            ax[1].plot(ms, func[zi,:], linewidth=1, label=('$z=${}'.format(fmt(zz)) if zz in zs[:count//count] else None), color='r')
        elif zz in zs[-count:]:
            ax[1].plot(ms, func[zi,:], linewidth=1, label=('$z=${}'.format(fmt(zz)) if zz in zs[-count//count:] else None), color='b')

    ax[0].set_xlabel('z')
    ax[1].set_xlabel('m')
    for axx in ax:
        axx.set_yscale('log')
        axx.set_xscale('log')
        axx.set_ylabel(title)
        axx.legend(); axx.grid()
    return ax

def plot_ucosth(ms, zs, angs, ucosth, prob, title, count=10):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    nMs, nZs = len(ms), len(zs)
    for mi, mm in enumerate(ms):
        if mm in ms[:count]:
            lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==0 else None)
            zi = 0
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], ms=3, marker='o', label=lab(zi), color='k')
            zi = nZs-1
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], ms=3, marker='o', label=lab(zi), color='r')
        elif mm in ms[-count:]:
            lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==len(ms)-1 else None)
            zi = 0
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], ms=3, marker='o', label=lab(zi), color='g')
            zi = nZs-1
            ax[0].plot(angs[:,zi,mi], ucosth[:,zi,mi], ms=3, marker='o', label=lab(zi), color='b')

    for mi, mm in enumerate(ms):
        if mm in ms[:count]:
            lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==0 else None)
            zi = 0
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], ms=3, marker='*', label=lab(zi), color='k')
            zi = nZs-1
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], ms=3, marker='*', label=lab(zi), color='r')
        elif mm in ms[-count:]:
            lab = lambda zi: (r'$m=${}'.format(fmt(mm))+f', $z=%5.2f$'%(zs[zi]) if mi==len(ms)-1 else None)
            zi = 0
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], ms=3, marker='*', label=lab(zi), color='g')
            zi = nZs-1
            ax[1].plot(angs[:,zi,mi], ucosth[:,zi,mi]*prob[zi,mi], ms=3, marker='*', label=lab(zi), color='b')

    for axx in ax:
        axx.set_yscale('log'); axx.set_xscale('log')
        axx.set_ylabel(r'$u(\cos(\theta))$'); axx.set_xlabel(r'$\theta$')
        axx.legend(); axx.grid()
    return ax

#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

def annot_max(xmax, ymax, lab, col, xcap, mind, ax):
    text = lab
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec=col, lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60",color=col)
    kw = dict(xycoords='data',textcoords="data",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(xmax-0.01, ymax-5e6), **kw)
