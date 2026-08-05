import numpy as np, json, sys, os
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import eigvals
from verify_revision import P0, normal_equilibrium, jacobian, rhs_full, simulate_cycle

_, bcrit=normal_equilibrium(); a,c=P0[4],P0[5]
yscale=np.array([0.1,0.01,0.01,0.01,0.001,1.0]); Tscale=250.; qdot_scale=1e-5

def integrate_aug(x,T,b,rtol=2e-9,atol=2e-11):
    z0=np.r_[x,np.eye(6).ravel()]
    def aug(t,z):
        y=z[:6]; M=z[6:].reshape(6,6)
        reg='N' if a*y[3] < c*(b-y[4]) else 'C'
        J=jacobian(y,b,reg)
        return np.r_[rhs_full(t,y,b),(J@M).ravel()]
    sol=solve_ivp(aug,(0,T),z0,method='DOP853',rtol=rtol,atol=atol,max_step=.3,dense_output=False)
    ze=sol.y[:,-1]
    return ze[:6],ze[6:].reshape(6,6),sol

def solve_b(b,z0):
    cache={}
    def eval(z):
        key=tuple(np.round(z,14))
        if key in cache:return cache[key]
        x=z[:6]*yscale;T=z[6]*Tscale
        if T<50 or T>1000:
            R=np.ones(7)*1e3;J=np.eye(7);cache[key]=(R,J,None);return R,J,None
        ye,M,sol=integrate_aug(x,T,b)
        R=np.r_[(ye-x)/yscale,rhs_full(0,x,b)[3]/qdot_scale]
        J=np.zeros((7,7));D=np.diag(yscale);Di=np.diag(1/yscale)
        J[:6,:6]=Di@(M-np.eye(6))@D
        J[:6,6]=Di@rhs_full(T,ye,b)*Tscale
        reg='N' if a*x[3] < c*(b-x[4]) else 'C'
        J[6,:6]=jacobian(x,b,reg)[3,:]*yscale/qdot_scale
        cache[key]=(R,J,(ye,M,sol));return R,J,(ye,M,sol)
    res=root(lambda z:eval(z)[0],z0,jac=lambda z:eval(z)[1],method='hybr',options={'xtol':1e-9,'maxfev':100})
    R,J,extra=eval(res.x)
    return res,R,extra

def stats(x,T,b,M):
    sol=solve_ivp(lambda t,y:rhs_full(t,y,b),(0,T),x,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.15,dense_output=True)
    tt=np.linspace(0,T,4001);yy=sol.sol(tt);psi=a*yy[3]-c*(b-yy[4])
    vals=eigvals(M);idx=int(np.argmin(np.abs(vals-1)));rho=float(np.max(np.abs(np.delete(vals,idx))))
    return {'b':float(b),'T':float(T),'qmin':float(yy[3].min()),'qmax':float(yy[3].max()),'ampq':float(np.ptp(yy[3])),
            'hmin':float(yy[4].min()),'hmax':float(yy[4].max()),'amph':float(np.ptp(yy[4])),
            'psimin':float(psi.min()),'psimax':float(psi.max()),'fracC':float(np.mean(psi>=0)),
            'rho':rho,'multipliers':[[float(v.real),float(v.imag)] for v in vals]},tt,yy

mode=sys.argv[1]
outfile=sys.argv[2]
if mode=='init':
    b=.5*bcrit
    eq,T,frac,tt,yy,psi=simulate_cycle(b,bcrit)
    j=int(np.argmax(yy[3]));x=yy[:,j];z=np.r_[x/yscale,T/Tscale]
    res,R,extra=solve_b(b,z);x=res.x[:6]*yscale;T=res.x[6]*Tscale;ye,M,sol=extra
    st,tt,yy=stats(x,T,b,M);st['resnorm']=float(np.max(np.abs(R)));st['success']=bool(res.success)
    rows=[st]
    np.savez('fixed_b_last.npz',x=x,T=T,b=b,z=res.x)
    json.dump(rows,open(outfile,'w'),indent=2)
    print('init',res.success,st)
else:
    targets=np.array([float(v) for v in sys.argv[3].split(',')])
    D=np.load('fixed_b_last.npz');x=D['x'];T=float(D['T']);z=np.r_[x/yscale,T/Tscale]
    rows=json.load(open(outfile)) if os.path.exists(outfile) else []
    for b in targets:
        res,R,extra=solve_b(float(b),z);x=res.x[:6]*yscale;T=res.x[6]*Tscale
        inf=float(np.max(np.abs(R)))
        if (not res.success) or inf>2e-6:
            print('FAIL',b,res.success,res.message,inf,T);break
        ye,M,sol=extra;st,tt,yy=stats(x,T,float(b),M);st['resnorm']=inf;st['success']=True
        rows.append(st);z=res.x
        np.savez('fixed_b_last.npz',x=x,T=T,b=b,z=z)
        json.dump(rows,open(outfile,'w'),indent=2)
        print('b',b,'T',T,'rho',st['rho'],'qmax',st['qmax'],'psi',st['psimin'],st['psimax'],'res',inf)
