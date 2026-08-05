import numpy as np, json, sys, os
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import eigvals
from verify_revision import P0, normal_equilibrium, jacobian, rhs_full

_,bcrit=normal_equilibrium(); a,c=P0[4],P0[5]
yscale=np.array([0.1,0.01,0.01,0.01,0.001,1.0]);Tscale=220.;bscale=0.002; qdot_scale=1e-5;qscale=0.01

def f_b(y,b):
 out=np.zeros(6)
 if a*y[3]>=c*(b-y[4]):out[3]=-c;out[4]=c
 return out

def integrate_aug(x,T,b,rtol=2e-9,atol=2e-11):
 z0=np.r_[x,np.eye(6).ravel(),np.zeros(6)]
 def aug(t,z):
  y=z[:6];M=z[6:42].reshape(6,6);sb=z[42:]
  reg='N' if a*y[3]<c*(b-y[4]) else 'C';J=jacobian(y,b,reg)
  return np.r_[rhs_full(t,y,b),(J@M).ravel(),J@sb+f_b(y,b)]
 sol=solve_ivp(aug,(0,T),z0,method='DOP853',rtol=rtol,atol=atol,max_step=.3)
 ze=sol.y[:,-1];return ze[:6],ze[6:42].reshape(6,6),ze[42:],sol

class Corrector:
 def __init__(self,Q):self.Q=Q;self.cache={}
 def unpack(self,z):return z[:6]*yscale,z[6]*Tscale,z[7]*bscale
 def eval(self,z):
  key=tuple(np.round(z,14))
  if key in self.cache:return self.cache[key]
  x,T,b=self.unpack(z)
  if not(30<T<1000 and 1e-7<b<0.01):
   R=np.ones(8)*1e3;J=np.eye(8);self.cache[key]=(R,J,None);return R,J,None
  ye,M,sb,sol=integrate_aug(x,T,b)
  R=np.r_[(ye-x)/yscale,rhs_full(0,x,b)[3]/qdot_scale,(x[3]-self.Q)/qscale]
  Jall=np.zeros((8,8));D=np.diag(yscale);Di=np.diag(1/yscale)
  Jall[:6,:6]=Di@(M-np.eye(6))@D
  Jall[:6,6]=Di@rhs_full(T,ye,b)*Tscale
  Jall[:6,7]=Di@sb*bscale
  reg='N' if a*x[3]<c*(b-x[4]) else 'C';Jx=jacobian(x,b,reg)
  Jall[6,:6]=Jx[3,:]*yscale/qdot_scale;Jall[6,7]=f_b(x,b)[3]*bscale/qdot_scale
  Jall[7,3]=yscale[3]/qscale
  self.cache[key]=(R,Jall,(ye,M,sb,sol));return R,Jall,(ye,M,sb,sol)
 def fun(self,z):return self.eval(z)[0]
 def jac(self,z):return self.eval(z)[1]

def solve_Q(Q,z0):
 C=Corrector(Q);res=root(C.fun,z0,jac=C.jac,method='hybr',options={'xtol':1e-9,'maxfev':120});R,J,extra=C.eval(res.x);return res,R,extra

def stats(x,T,b,M):
 sol=solve_ivp(lambda t,y:rhs_full(t,y,b),(0,T),x,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.15,dense_output=True)
 tt=np.linspace(0,T,4001);yy=sol.sol(tt);psi=a*yy[3]-c*(b-yy[4])
 vals=eigvals(M);idx=int(np.argmin(np.abs(vals-1)));rho=float(np.max(np.abs(np.delete(vals,idx))))
 return {'b':float(b),'T':float(T),'qmin':float(yy[3].min()),'qmax':float(yy[3].max()),'ampq':float(np.ptp(yy[3])),
 'hmin':float(yy[4].min()),'hmax':float(yy[4].max()),'amph':float(np.ptp(yy[4])),'psimin':float(psi.min()),'psimax':float(psi.max()),
 'fracC':float(np.mean(psi>=0)),'rho':rho,'multipliers':[[float(v.real),float(v.imag)] for v in vals]}

outfile=sys.argv[1];targets=np.array([float(v) for v in sys.argv[2].split(',')])
if os.path.exists('qmax_last.npz'):
 D=np.load('qmax_last.npz');x=D['x'];T=float(D['T']);b=float(D['b'])
else:
 D=np.load('fixed_b_last.npz');x=D['x'];T=float(D['T']);b=float(D['b'])
z=np.r_[x/yscale,T/Tscale,b/bscale]
rows=json.load(open(outfile)) if os.path.exists(outfile) else []
for Q in targets:
 res,R,extra=solve_Q(Q,z);x=res.x[:6]*yscale;T=res.x[6]*Tscale;b=res.x[7]*bscale;inf=float(np.max(np.abs(R)))
 if (not res.success) or inf>2e-6:
  print('FAIL',Q,res.success,res.message,inf,b,T);break
 ye,M,sb,sol=extra;st=stats(x,T,b,M);st.update({'Q_target':float(Q),'resnorm':inf})
 rows.append(st);z=res.x
 np.savez('qmax_last.npz',x=x,T=T,b=b,z=z,Q=Q);json.dump(rows,open(outfile,'w'),indent=2)
 print('Q',Q,'b',b,'T',T,'rho',st['rho'],'psi',st['psimin'],st['psimax'],'res',inf)
