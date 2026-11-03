import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---- 1. Orbital elements (MPC solution as listed on in-the-sky.org/data/object.php?id=0001P)
a  = 17.86          # au
e  = 0.967941
inc= np.radians(162.16)
Om = np.radians(59.52)    # longitude of ascending node (J2000 ecliptic)
w  = np.radians(112.42)   # argument of perihelion
M0 = np.radians(189.56)   # mean anomaly at epoch
t0 = datetime(2025,11,21) # epoch
P  = a**1.5               # years (Kepler's 3rd law)  ~75.5 yr
n  = 2*np.pi/P            # rad / yr
YR = 365.25

# ---- 2. Kepler's equation  M = E - e sin E  (Newton-Raphson)
def solve_kepler(M, e, tol=1e-12):
    M = np.mod(M, 2*np.pi)
    E = np.where(e < 0.8, M, np.pi)           # robust start for high e
    for _ in range(100):
        dE = (E - e*np.sin(E) - M)/(1 - e*np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < tol: break
    return E

# ---- 3. position in HE (heliocentric ecliptic) frame
def position(dt_years_from_epoch):
    M = M0 + n*dt_years_from_epoch
    E = solve_kepler(M, e)
    x = a*(np.cos(E) - e)                     # perifocal frame
    y = a*np.sqrt(1-e**2)*np.sin(E)
    cO,sO,ci,si,cw,sw = np.cos(Om),np.sin(Om),np.cos(inc),np.sin(inc),np.cos(w),np.sin(w)
    X = (cO*cw - sO*sw*ci)*x + (-cO*sw - sO*cw*ci)*y
    Y = (sO*cw + cO*sw*ci)*x + (-sO*sw + cO*cw*ci)*y
    Z = (sw*si)*x + (cw*si)*y
    return np.array([X,Y,Z]).T

def yrs(d): return (d - t0).total_seconds()/86400/YR

# perihelion date from M = 2*pi
tp = t0 + timedelta(days=((2*np.pi - M0) % (2*np.pi))/n*YR)
print("Perihelion (2-body):", tp.date(), " q =", a*(1-e), "au, P =", P, "yr")

# ---- 4. sampling: yearly for 40 yr + dense near perihelion
start = datetime(2026,10,29)
yearly = [start.replace(year=start.year+k) for k in range(41)]
dense  = [tp + timedelta(days=d) for d in
          list(range(-365,-90,30)) + list(range(-90,91,5)) + list(range(120,366,30))]
rows = [(d,"yearly") for d in yearly] + [(d,"near-perihelion") for d in dense]
df = pd.DataFrame(rows, columns=["date","type"]).sort_values("date").reset_index(drop=True)
t = np.array([yrs(d) for d in df.date])
r = position(t)
df[["X_au","Y_au","Z_au"]] = r
df["r_au"] = np.linalg.norm(r,axis=1)
df.to_csv("halley_positions.csv", index=False, float_format="%.5f")
print(df[df.type=="yearly"].to_string(index=False, float_format=lambda v:f"{v:8.3f}"))

# ---- 5. plot
full = position(t0_ := np.linspace(0,P,4000)+0)  # one full orbit
fig = plt.figure(figsize=(14,6.5))
ax = fig.add_subplot(121, projection="3d")
ax.plot(*full.T, color="0.7", lw=0.8, label="full orbit")
Y = df[df.type=="yearly"]; D = df[df.type=="near-perihelion"]
ax.scatter(Y.X_au,Y.Y_au,Y.Z_au,s=18,c="tab:blue",label="yearly")
ax.scatter(D.X_au,D.Y_au,D.Z_au,s=8,c="tab:red",label="dense near perihelion")
th=np.linspace(0,2*np.pi,200)
for rr,c,l in [(1,"g","Earth"),(5.2,"orange","Jupiter"),(9.58,"goldenrod","Saturn"),(19.2,"c","Uranus"),(30.07,"b","Neptune")]:
    ax.plot(rr*np.cos(th),rr*np.sin(th),0,color=c,lw=0.6,label=l)
ax.scatter([0],[0],[0],c="gold",s=80,edgecolors="k",label="Sun")
ax.set_xlabel("X (au)");ax.set_ylabel("Y (au)");ax.set_zlabel("Z (au)")
ax.set_title("1P/Halley, heliocentric ecliptic (HE) frame, 2026–2066")
ax.legend(fontsize=7,loc="upper left"); ax.view_init(22,-50)
ax2 = fig.add_subplot(122)
tt = np.linspace(yrs(start),yrs(yearly[-1]),3000)
dd = [t0+timedelta(days=x*YR) for x in tt]
ax2.plot(dd,np.linalg.norm(position(tt),axis=1))
ax2.scatter(Y.date,Y.r_au,c="tab:blue",s=15); ax2.scatter(D.date,D.r_au,c="tab:red",s=6)
ax2.set_ylabel("distance from Sun (au)"); ax2.set_title("Heliocentric distance"); ax2.grid(alpha=.3)
plt.tight_layout(); plt.savefig("halley_trajectory.png",dpi=150)