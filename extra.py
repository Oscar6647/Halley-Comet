import numpy as np, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
import Halley as H   # same elements/solver as halley.py

MU = 4*np.pi**2            # au^3/yr^2
KMS = 4.74047              # au/yr -> km/s

# ---------- speed profile (vis-viva)
def speed(r): return np.sqrt(MU*(2/r - 1/H.a))*KMS
tt = np.linspace(H.yrs(datetime(2026,10,29)), H.yrs(datetime(2066,10,29)), 4000)
rr = np.linalg.norm(H.position(tt),axis=1)
dates = [H.t0+timedelta(days=x*365.25) for x in tt]
fig,ax = plt.subplots(figsize=(9,4)); ax.plot(dates,speed(rr)); ax.set_ylabel("speed (km/s)")
ax.set_title("Halley's heliocentric speed: crawl at aphelion, ~55 km/s at perihelion"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("halley_speed.png",dpi=150); plt.close()
print("v_peri km/s:", speed(H.a*(1-H.e)), " v_now:", speed(rr[0]))

# ---------- Earth (low-precision Keplerian, J2000 ecliptic)
def earth(date):
    T = ((date - datetime(2000,1,1,12)).total_seconds()/86400)/36525
    L = np.radians(100.46435 + 35999.37285*T); wb = np.radians(102.94719); e=0.016709
    M = L-wb; E = H.solve_kepler(np.array(M),e)
    x = np.cos(E)-e; y=np.sqrt(1-e*e)*np.sin(E); nu=np.arctan2(y,x); r=1.00000261*(1-e*np.cos(E))
    return np.array([r*np.cos(nu+wb), r*np.sin(nu+wb), 0.0])
def earth_state(date):
    h=0.5/365.25; d=timedelta(days=0.5)
    return earth(date), (earth(date+d)-earth(date-d))/(2*h)
def halley_state(date):
    t=H.yrs(date); h=0.5/365.25
    return H.position(np.array([t]))[0], (H.position(np.array([t+h]))[0]-H.position(np.array([t-h]))[0])/(2*h)

# ---------- Lambert (universal variables, Bate-Mueller-White)
def stumpff(z):
    if z>1e-8: s=np.sqrt(z); return (s-np.sin(s))/s**3,(1-np.cos(s))/z
    if z<-1e-8: s=np.sqrt(-z); return (np.sinh(s)-s)/s**3,(np.cosh(s)-1)/(-z)
    return 1/6,0.5
def lambert(r1,r2,dt):
    n1,n2=np.linalg.norm(r1),np.linalg.norm(r2)
    c=np.dot(r1,r2)/(n1*n2); dnu=np.arccos(np.clip(c,-1,1))
    if np.cross(r1,r2)[2]<0: dnu=2*np.pi-dnu          # prograde transfer
    A=np.sin(dnu)*np.sqrt(n1*n2/(1-np.cos(dnu)))
    def tof(z):
        S,C=stumpff(z); y=n1+n2+A*(z*S-1)/np.sqrt(C)
        if y<0: return -1e9
        return ((y/C)**1.5*S + A*np.sqrt(y))/np.sqrt(MU)-dt
    try: z=brentq(tof,-30,4*np.pi**2-1e-3,xtol=1e-10)
    except ValueError: return None
    S,C=stumpff(z); y=n1+n2+A*(z*S-1)/np.sqrt(C)
    f=1-y/n1; g=A*np.sqrt(y/MU); gd=1-y/n2
    return (r2-f*r1)/g, (gd*r2-r1)/g

# ---------- porkchop: launch vs arrival
peri = H.tp
L0 = peri - timedelta(days=420); A0 = peri - timedelta(days=100)
launch = [L0+timedelta(days=5*i) for i in range(0,70)]
arrive = [A0+timedelta(days=4*j) for j in range(0,80)]
C3 = np.full((len(arrive),len(launch)),np.nan); VINF=C3.copy()
for j,ta in enumerate(arrive):
    rh,vh = halley_state(ta)
    for i,tl in enumerate(launch):
        dt=(ta-tl).days/365.25
        if dt<0.15: continue
        re,ve = earth_state(tl)
        s=lambert(re,rh,dt)
        if s is None: continue
        v1,v2=s
        C3[j,i]=(np.linalg.norm(v1-ve)*KMS)**2
        VINF[j,i]=np.linalg.norm(v2-vh)*KMS
k=np.nanargmin(C3); j,i=np.unravel_index(k,C3.shape)
print(f"Best: launch {launch[i].date()} arrive {arrive[j].date()} TOF {(arrive[j]-launch[i]).days} d  C3={C3[j,i]:.1f} km2/s2  v_inf_arr={VINF[j,i]:.1f} km/s")
Cp=np.where(C3<400,C3,np.nan)
fig,ax=plt.subplots(figsize=(9,6))
cs=ax.contourf(launch,arrive,Cp,levels=np.arange(0,401,20),cmap="viridis_r")
ax.contour(launch,arrive,Cp,levels=[50,100,200],colors="w",linewidths=.6)
ax.plot(launch[i],arrive[j],"r*",ms=14,label=f"min C3 = {C3[j,i]:.0f} km²/s²")
plt.colorbar(cs,label="launch C3 (km²/s²)"); ax.set_xlabel("launch date"); ax.set_ylabel("Halley arrival date")
ax.set_title("Porkchop plot: Earth → Halley flyby near 2061 perihelion"); ax.legend(); fig.autofmt_xdate()
plt.tight_layout(); plt.savefig("halley_porkchop.png",dpi=150); plt.close()

# ---------- transfer arc for best case
tl,ta=launch[i],arrive[j]; re,ve=earth_state(tl); rh,vh=halley_state(ta)
v1,_=lambert(re,rh,(ta-tl).days/365.25)
sol=solve_ivp(lambda t,s:np.r_[s[3:],-MU*s[:3]/np.linalg.norm(s[:3])**3],[0,(ta-tl).days/365.25],np.r_[re,v1],rtol=1e-10,atol=1e-12,dense_output=True)
arc=sol.sol(np.linspace(0,sol.t[-1],200))[:3].T
t=np.array([H.yrs(peri+timedelta(days=d)) for d in range(-700,500,5)]); hp=H.position(t)
fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection="3d")
th=np.linspace(0,2*np.pi,200)
ax.plot(np.cos(th),np.sin(th),0,"g",lw=.8,label="Earth orbit")
ax.plot(*hp.T,"0.6",lw=1,label="Halley (2060–2062)")
ax.plot(*arc.T,"r",lw=2,label="spacecraft")
ax.scatter(*rh,c="tab:red",s=40); ax.scatter(*re,c="g",s=40); ax.scatter(0,0,0,c="gold",s=100,edgecolors="k")
ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_zlim(-2,2);ax.legend(); ax.set_xlabel("X (au)");ax.set_ylabel("Y (au)");ax.set_zlabel("Z (au)")
ax.set_title(f"Flyby: launch {tl.date()}, intercept {ta.date()}"); plt.tight_layout(); plt.savefig("halley_mission.png",dpi=150); plt.close()

# ---------- data for interactive viewer
full=H.position(np.linspace(0,H.P,600)+0)
ts=np.linspace(H.yrs(datetime(2026,10,29)),H.yrs(datetime(2066,10,29)),241)
pos=H.position(ts); rs=np.linalg.norm(pos,axis=1)
json.dump({"orbit":full.round(3).tolist(),"pos":pos.round(3).tolist(),"r":rs.round(2).tolist(),"v":speed(rs).round(1).tolist(),
 "date":[(H.t0+timedelta(days=x*365.25)).strftime("%Y-%m-%d") for x in ts],"arc":arc.round(3).tolist()},open("viewer_data.json","w"))