from pathlib import Path
import json, numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from verify_revision import P0, R0_value, normal_equilibrium, transient_threshold, capacity_equilibria, jacobian

OUT=Path(__file__).resolve().parent

def load(name): return json.load(open(OUT/name))
def load_parts(pattern):
    data=[]
    for path in sorted(OUT.glob(pattern)):
        data.extend(json.load(open(path)))
    return data
# periodic branches
uh=load('unstable_branch_hopf.json')+load('unstable_branch_hopf_tail.json')
sd=load('stable_branch_down.json')
su=load('stable_branch_up.json')
qm=load_parts('qmax_branch_parts/qmax_branch_*.json')
# dedupe/sort plotting segments individually
fig,axs=plt.subplots(1,2,figsize=(10.8,4.3))
ax=axs[0]
for data,style,label in [(uh,'--','unstable periodic orbit'),(sd,'-','stable periodic orbit'),(su,'-',None),(qm,'--',None)]:
    ax.plot([x['b'] for x in data],[x['ampq'] for x in data],style,lw=1.8,label=label)
# equilibrium branch stability proxy at amplitude zero
_,bc=normal_equilibrium(); bH=2.32155208e-4
ax.plot([4e-5,bH],[0,0],'-',lw=2,label='stable equilibrium')
ax.plot([bH,bc],[0,0],'--',lw=2,label='unstable equilibrium')
ax.plot([bc,2.25e-3],[0,0],'-',lw=2)
# numerical landmarks
bL=8.6241e-5; bU=max(x['b'] for x in qm)
ax.plot(bH,0,'^',ms=7,label='Hopf $H$')
ax.plot([bL,bU],[0.00973,0.00444],'s',ms=6,label='fold of cycles $LPC$')
ax.axvline(bc,ls=':',lw=1.1)
ax.text(bc*1.01,0.0115,r'$b_{\rm crit}$',rotation=90,va='top',fontsize=8)
ax.set_xscale('log'); ax.set_xlim(5e-5,2.4e-3); ax.set_ylim(-0.00045,0.0142)
ax.set_xlabel('hospital capacity $b$'); ax.set_ylabel(r'periodic-orbit amplitude $\max q-\min q$')
ax.legend(fontsize=7,loc='upper right',ncol=2)
ax.set_title('(a) global periodic-orbit structure')

ax=axs[1]
# Floquet factor excluding trivial 1
for data,style,label in [(uh,'--','unstable branch'),(sd,'-','stable branch'),(su,'-',None),(qm,'--',None)]:
    xs=[]; ys=[]
    for x in data:
        if x.get('rho') is not None and np.isfinite(x['rho']): xs.append(x['b']); ys.append(x['rho'])
    ax.plot(xs,ys,style,lw=1.8,label=label)
ax.axhline(1,ls=':',lw=1)
ax.axvline(bH,ls=':',lw=1); ax.axvline(bL,ls=':',lw=1); ax.axvline(bU,ls=':',lw=1)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(5e-5,2.4e-3); ax.set_ylim(0.45,6)
ax.set_xlabel('hospital capacity $b$'); ax.set_ylabel('largest nontrivial Floquet multiplier modulus')
ax.set_title('(b) cycle stability')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'fig_global_bifurcation.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig_global_bifurcation.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# augmented two-parameter threshold/stability atlas
Bvals=np.linspace(0.75,6.0,28); bcvals=np.full_like(Bvals,np.nan); btvals=np.full_like(Bvals,np.nan); bhvals=np.full_like(Bvals,np.nan)
for j,B in enumerate(Bvals):
    p=P0.copy(); p[0]=B
    if R0_value(p)<=1: continue
    _,bcj=normal_equilibrium(p); bcvals[j]=bcj
    btvals[j]=transient_threshold(p,160)[0]
    def alphaC(b):
        roots=capacity_equilibria(b,p)
        if not roots: return np.nan
        vals=np.linalg.eigvals(jacobian(roots[0],b,'C',p))
        return np.max(vals.real)
    lo=max(1e-10,bcj*1e-4); hi=bcj*(1-1e-8)
    grid=np.geomspace(lo,hi,60); vals=np.array([alphaC(x) for x in grid])
    idx=np.where(vals[:-1]*vals[1:]<0)[0]
    if len(idx):
        k=idx[-1]
        bhvals[j]=brentq(alphaC,grid[k],grid[k+1],xtol=1e-13)

fig,ax=plt.subplots(figsize=(6.2,4.7))
ax.plot(Bvals,bcvals,lw=1.8,label=r'$b_{\rm crit}$')
ax.plot(Bvals,btvals,lw=1.8,label=r'$b_{\rm trans}$')
ax.plot(Bvals,bhvals,lw=1.8,ls='--',label=r'$b_H$ (smooth Hopf)')
ax.fill_between(Bvals,bcvals,btvals,where=np.isfinite(btvals),alpha=.18,label='transient-overload band')
ax.fill_between(Bvals,bhvals,bcvals,where=np.isfinite(bhvals),alpha=.14,label='unstable capacity equilibrium')
Binv=(P0[2]+P0[1])/P0[2]/(1+P0[11]*P0[3]/(P0[4]+P0[6]))
ax.axvline(Binv,ls=':',lw=1.1,label=r'$R_0=1$')
ax.set_xlabel(r'transmission parameter $\mathcal{B}$'); ax.set_ylabel('hospital capacity $b$'); ax.set_ylim(bottom=0)
ax.legend(fontsize=7.5,loc='upper left'); fig.tight_layout(); fig.savefig(OUT/'fig_bifurcation_atlas.pdf',bbox_inches='tight'); fig.savefig(OUT/'fig_bifurcation_atlas.png',dpi=220,bbox_inches='tight'); plt.close(fig)

json.dump({'bLPC_lower_approx':bL,'bLPC_upper_approx':bU,'bH':bH,'bcrit':bc},open(OUT/'global_bifurcation_summary.json','w'),indent=2)
print('created figures', bL,bU)
