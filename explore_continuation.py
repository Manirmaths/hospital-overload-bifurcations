import numpy as np
from scipy.integrate import solve_bvp
from scipy.linalg import eig
from verify_revision import normal_equilibrium, capacity_equilibria, jacobian, find_hopf, rhs_full

_, bcrit=normal_equilibrium()
bH,yH,evH,trans=find_hopf(bcrit)
J=jacobian(yH,bH,'C')
vals,vecs=eig(J)
k=np.argmin(np.abs(vals-1j*0.03629529))
lam=vals[k]; q=vecs[:,k]
q=q/np.max(np.abs(q))
print('bcrit,bH,lam',bcrit,bH,lam)

def rhs_vec(y,b):
    out=np.zeros_like(y)
    for j in range(y.shape[1]): out[:,j]=rhs_full(0,y[:,j],b)
    return out

def solve_orbit_fixed_b(b, y_guess, T_guess, s_mesh=None, ref=None, tol=1e-8, max_nodes=3000):
    if s_mesh is None: s_mesh=np.linspace(0,1,y_guess.shape[1])
    if ref is None:
        ref_y0=y_guess[:,0].copy()
        ref_tan=T_guess*rhs_full(0,ref_y0,b)
        if np.linalg.norm(ref_tan)<1e-12:
            ref_tan=(y_guess[:,1]-y_guess[:,-2])/(s_mesh[1]+1-s_mesh[-2])
    else:
        ref_y0,ref_tan=ref
    ref_tan=np.asarray(ref_tan)
    def fun(s,y,par):
        T=par[0]
        return T*rhs_vec(y,b)
    def bc(ya,yb,par):
        return np.r_[ya-yb, np.dot(ya-ref_y0,ref_tan)]
    return solve_bvp(fun,bc,s_mesh,y_guess,p=np.array([T_guess]),tol=tol,max_nodes=max_nodes,verbose=0)

s=np.linspace(0,1,301)
for delta in [1e-7,3e-7,1e-6,3e-6,1e-5,3e-5]:
    b=bH-delta
    eq=capacity_equilibria(b)[0]
    mu=trans*(b-bH)
    A=np.sqrt(max(1e-16,-mu/42.441786))
    print('delta',delta,'A nf',A)
    for mult in [0.2,0.5,1,2,5,10,20,50]:
        amp=mult*A
        yg=eq[:,None] + amp*(np.real(q)[:,None]*np.cos(2*np.pi*s)[None,:]-np.imag(q)[:,None]*np.sin(2*np.pi*s)[None,:])
        T=2*np.pi/lam.imag
        sol=solve_orbit_fixed_b(b,yg,T,s,tol=1e-7,max_nodes=5000)
        amp_sol=np.max(sol.y[3])-np.min(sol.y[3])
        print(' mult',mult,'success',sol.success,'status',sol.status,'T',sol.p[0],'qrange',amp_sol,'nodes',sol.x.size,'msg',sol.message)
        if sol.success and amp_sol>1e-9:
            np.savez('first_orbit.npz',x=sol.x,y=sol.y,T=sol.p[0],b=b)
            raise SystemExit
print('failed')
