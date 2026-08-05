import numpy as np, json, time
from scipy.integrate import solve_ivp
from scipy.optimize import root, brentq
from scipy.linalg import eigvals
from verify_revision import P0, normal_equilibrium, jacobian, find_hopf, rhs_full

B,m,eta,pi,a,c,chiQ,chiH,rhoI,rhoQ,rhoH,theta=P0
_, bcrit=normal_equilibrium(); bH,yH,evH,trans=find_hopf(bcrit)
yscale=np.array([0.1,0.01,0.01,0.01,0.001,1.0])
Tscale=200.0; bscale=bH; qdot_scale=1e-5; amp_scale=0.01

def eqC_fast(b):
    L=(eta+m)/(m*eta)
    h=c/(c+chiH)*b
    A=c*chiH/(c+chiH)*b
    imin=A*(a+chiQ)/(a*pi)
    imax=(1.0/L)*(1-1e-12)
    def q(i): return (pi*i-A)/chiQ
    def s(i): return 1-L*i
    def r(i): return (rhoI*i+rhoQ*q(i)+rhoH*h)/m
    def n(i): return s(i)+i/eta+i+q(i)+h+r(i)
    def f(i): return B*s(i)*(i+theta*q(i))-(eta+m)/eta*i*n(i)
    if f(imin)==0: ii=imin
    else: ii=brentq(f,imin,imax,xtol=5e-15,rtol=5e-14)
    return np.array([s(ii),ii/eta,ii,q(ii),h,r(ii)])

def deq_q_db(b):
    d=max(1e-10,abs(b)*2e-5)
    return (eqC_fast(b+d)[3]-eqC_fast(b-d)[3])/(2*d)

def f_b(y,b):
    out=np.zeros(6)
    if a*y[3] >= c*(b-y[4]):
        out[3]=-c; out[4]=c
    return out

def integrate_aug(x,T,b,rtol=2e-9,atol=2e-11):
    z0=np.r_[x,np.eye(6).ravel(),np.zeros(6)]
    def aug(t,z):
        y=z[:6]; M=z[6:42].reshape(6,6); sb=z[42:48]
        reg='N' if a*y[3] < c*(b-y[4]) else 'C'
        J=jacobian(y,b,reg)
        return np.r_[rhs_full(t,y,b),(J@M).ravel(),J@sb+f_b(y,b)]
    sol=solve_ivp(aug,(0,T),z0,method='DOP853',rtol=rtol,atol=atol,max_step=.4)
    ze=sol.y[:,-1]
    return ze[:6],ze[6:42].reshape(6,6),ze[42:48],sol

class Corrector:
    def __init__(self,A): self.A=A; self.cache={}
    def unpack(self,z): return z[:6]*yscale,z[6]*Tscale,z[7]*bscale
    def eval(self,z,need_jac=True):
        key=tuple(np.round(z,14))
        if key in self.cache: return self.cache[key]
        x,T,b=self.unpack(z)
        if not (20<T<1000 and 1e-8<b<bcrit):
            R=np.ones(8)*1e3; J=np.eye(8); self.cache[key]=(R,J,None); return R,J,None
        eq=eqC_fast(b)
        ye,M,sb,sol=integrate_aug(x,T,b)
        per=(ye-x)/yscale
        fx=rhs_full(0,x,b); Jx=jacobian(x,b,'N' if a*x[3]<c*(b-x[4]) else 'C')
        qdot=fx[3]/qdot_scale
        amp=(x[3]-eq[3]-self.A)/amp_scale
        R=np.r_[per,qdot,amp]
        JJ=np.zeros((8,8))
        Dinv=np.diag(1/yscale); D=np.diag(yscale)
        JJ[:6,:6]=Dinv@(M-np.eye(6))@D
        JJ[:6,6]=Dinv@rhs_full(T,ye,b)*Tscale
        JJ[:6,7]=Dinv@sb*bscale
        JJ[6,:6]=Jx[3,:]*yscale/qdot_scale
        JJ[6,7]=f_b(x,b)[3]*bscale/qdot_scale
        JJ[7,3]=yscale[3]/amp_scale
        JJ[7,7]=-deq_q_db(b)*bscale/amp_scale
        self.cache[key]=(R,JJ,(ye,M,sb,sol))
        return R,JJ,(ye,M,sb,sol)
    def fun(self,z): return self.eval(z)[0]
    def jac(self,z): return self.eval(z)[1]

def solve_A(A,z0):
    C=Corrector(A)
    res=root(C.fun,z0,jac=C.jac,method='hybr',options={'xtol':1e-9,'maxfev':100})
    R,J,extra=C.eval(res.x)
    return res,R,J,extra

def orbit_stats(x,T,b):
    sol=solve_ivp(lambda t,y: rhs_full(t,y,b),(0,T),x,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.15,dense_output=True)
    tt=np.linspace(0,T,4001); yy=sol.sol(tt); psi=a*yy[3]-c*(b-yy[4])
    return {'T':float(T),'b':float(b),'qmin':float(yy[3].min()),'qmax':float(yy[3].max()),
      'ampq':float(np.ptp(yy[3])),'hmin':float(yy[4].min()),'hmax':float(yy[4].max()),
      'amph':float(np.ptp(yy[4])),'psimin':float(psi.min()),'psimax':float(psi.max()),
      'fracC':float(np.mean(psi>=0))},tt,yy

import sys, os
name=sys.argv[1]
targets=np.array([float(v) for v in sys.argv[2].split(',')])
D=np.load('amplitude_last_fast.npz')
x=D['x'];T=float(D['T']);b=float(D['b']);z=np.r_[x/yscale,T/Tscale,b/bscale]
rows=[]
for k,A in enumerate(targets):
    res,R,J,extra=solve_A(float(A),z)
    x=res.x[:6]*yscale;T=res.x[6]*Tscale;b=res.x[7]*bscale
    inf=float(np.max(np.abs(R))); st,tt,yy=orbit_stats(x,T,b)
    if (not res.success) or inf>2e-6 or abs(st['qmax']-x[3])>1e-5:
        print('FAIL',k,A,res.success,res.message,inf,b,T);break
    ye,M,sb,sol=integrate_aug(x,T,b,rtol=8e-10,atol=8e-12)
    vals=eigvals(M);idx=int(np.argmin(np.abs(vals-1)));rho=float(np.max(np.abs(np.delete(vals,idx))))
    eq=eqC_fast(b); st.update({'A_target':float(A),'A_actual':float(x[3]-eq[3]),'resnorm':inf,'rho':rho,
       'multipliers':[[float(v.real),float(v.imag)] for v in vals]})
    rows.append(st); z=res.x
    np.savez('amplitude_last_fast.npz',z=z,x=x,T=T,b=b,A=A)
    with open(name,'w') as f:json.dump(rows,f,indent=2)
    print(k,'A',A,'b',b,'T',T,'rho',rho,'psi',st['psimin'],st['psimax'],'res',inf)
