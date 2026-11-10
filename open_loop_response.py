import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import control as ct

# --- identified parameters ---
K_servo, B_servo, I_rotor = 8.7831, 0.8701, 0.038967
r_p, m_f = 0.018, 0.09087
mu, phi, L, b = 0.22, np.radians(20), 0.105, 0.1456
eta = (1 - mu*np.tan(phi)) / (1 + mu*L/b)
k_s, b_s = 160.0, 0.0
I_eff = I_rotor + 2*m_f*r_p**2

MM = 25.4
CPI, T_poll, sigma_cnt = 12800, 5e-3, 0.0   # zero noise floor, per measurement
v_th = (MM/CPI)/T_poll

Theta_0 = 0.955          # rad, chosen to give F_ss ~= 2.71 N
t_release = 0.60          # s, well past the exact settling time (386 ms)
W, mu_o, m_obj = 3.00, 0.5, 0.05

G0_exact = K_servo*eta*k_s*r_p / (K_servo*eta + 2*k_s*r_p**2)
F_ss_exact = G0_exact*Theta_0   # closed-form; used as the reference, not a
                                # value read off the discretized trajectory

A = np.array([[0, 1], [-(K_servo+2*k_s*r_p**2/eta)/I_eff,
                        -(B_servo+2*b_s*r_p**2/eta)/I_eff]])
B = np.array([0, K_servo/I_eff])

dt = 2e-5
T_end = 0.8
N = int(T_end/dt)

z = np.zeros(2)
v_true = 0.0
accum, t_last_poll, v_meas = 0.0, 0.0, 0.0
count_size = MM/CPI

t_h, th_h, F_h, vt_h, vm_h = [], [], [], [], []
rng = np.random.default_rng(0)

for n in range(N):
    t_now = n*dt
    F_meas = max(0.0, k_s*r_p*z[0] + b_s*r_p*z[1])
    z = z + (A@z + B*Theta_0)*dt

    if t_now < t_release:
        v_true = 0.0
    else:
        net = W - 2*mu_o*F_meas
        if v_true > 0 or net > 0:
            v_true = max(0.0, v_true + (net/m_obj)*dt)
        else:
            v_true = 0.0

    accum += v_true*dt
    if t_now - t_last_poll >= T_poll:
        counts_true = accum/count_size
        counts_meas = max(0.0, round(counts_true + rng.normal(0, sigma_cnt)))
        v_meas = counts_meas*count_size/T_poll
        accum = 0.0
        t_last_poll = t_now

    t_h.append(t_now); th_h.append(z[0]); F_h.append(F_meas)
    vt_h.append(v_true); vm_h.append(v_meas)

t_h=np.array(t_h); th_h=np.array(th_h); F_h=np.array(F_h)
vt_h=np.array(vt_h); vm_h=np.array(vm_h)

F_ss = F_ss_exact
peak = F_h[t_h<t_release].max()
OS = 100*(peak-F_ss)/F_ss
tol=0.02*F_ss
mask=t_h<t_release
idx=np.where(np.abs(F_h[mask]-F_ss)>tol)[0]
t_s=t_h[idx[-1]] if len(idx) else 0
y10,y90=0.1*F_ss,0.9*F_ss
i10=np.where(F_h[mask]>=y10)[0][0]; i90=np.where(F_h[mask]>=y90)[0][0]
t_r=t_h[i90]-t_h[i10]
print(f"F_ss(exact)={F_ss:.4f}  F(t_release)={F_h[mask][-1]:.4f}  "
      f"residual gap={100*(F_h[mask][-1]-F_ss)/F_ss:+.3f}%")
print(f"rise={t_r*1000:.1f}ms OS={OS:.2f}% settle={t_s*1000:.1f}ms")
print(f"final slip velocity at t={T_end}s: {vt_h[-1]*1000:.1f} mm/s (still rising: open-loop never arrests it)")

t_ms = t_h * 1000
t_rel_ms = t_release * 1000
title_common = (
    "Open-Loop Response to a Step in Commanded Angle\n"
    r"$\theta_{cmd}$=" + f"{Theta_0:.3f} rad ({np.degrees(Theta_0):.1f}°), "
    f"$\\omega_n$=15.11 rad/s, $\\zeta$=0.738"
)

# --- Fig 1: pinion angle ---
fig1, ax1 = plt.subplots(figsize=(7.5, 3.2))
ax1.plot(t_ms, np.degrees(th_h), lw=2, color="#1a7f37")
ax1.axhline(np.degrees(Theta_0), color="k", ls="--", lw=1.0, label=r"$\theta_{cmd}$")
ax1.axvline(t_rel_ms, color="darkorange", ls="-.", lw=1.2, alpha=0.8)
ax1.text(t_rel_ms+8, ax1.get_ylim()[0]+2, "object released",
         fontsize=8, color="darkorange")
ax1.set_ylabel(r"Pinion angle $\theta$ (deg)")
ax1.set_xlabel("Time (ms)")
ax1.set_title(title_common)
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig("open_loop_angle.png", dpi=150)
print("saved open_loop_angle.png")
plt.close(fig1)

# --- Fig 2: grip force ---
fig2, ax2 = plt.subplots(figsize=(7.5, 3.2))
ax2.plot(t_ms, F_h, lw=2, color="#1f6feb")
ax2.axhline(F_ss, color="k", ls="--", lw=1.0, label=f"$F_{{ss}}$={F_ss:.2f} N")
ax2.axhline(W/(2*mu_o), color="#d1242f", ls=":", lw=1.2,
           label=f"needed to hold: $W/2\\mu_o$={W/(2*mu_o):.2f} N")
ax2.axvline(t_rel_ms, color="darkorange", ls="-.", lw=1.2, alpha=0.8)
ax2.set_ylabel("Grip force $F_{meas}$ (N)")
ax2.set_xlabel("Time (ms)")
ax2.set_title(title_common)
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("open_loop_force.png", dpi=150)
print("saved open_loop_force.png")
plt.close(fig2)

# --- Fig 3: slip velocity ---
fig3, ax3 = plt.subplots(figsize=(7.5, 3.2))
ax3.plot(t_ms, vm_h*1000, lw=1.4, color="#d1242f", alpha=0.85, label="measured")
ax3.plot(t_ms, vt_h*1000, lw=1.6, color="k", ls="--", label="true slip velocity")
ax3.axhline(v_th, color="gray", ls=":", lw=1.2,
           label=f"deadband $v_{{th}}$={v_th:.3f} mm/s (12800 CPI)")
ax3.axvline(t_rel_ms, color="darkorange", ls="-.", lw=1.2, alpha=0.8)
ax3.set_ylabel("Slip velocity (mm/s)")
ax3.set_xlabel("Time (ms)")
ax3.set_title(title_common)
ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

axins = ax3.inset_axes([0.32, 0.40, 0.42, 0.55])
mask2 = (t_h >= t_release-0.01) & (t_h <= t_release+0.05)
axins.plot(t_h[mask2]*1000, vm_h[mask2]*1000, color="#d1242f", lw=1.2, drawstyle="steps-post")
axins.plot(t_h[mask2]*1000, vt_h[mask2]*1000, color="k", lw=1.4, ls="--")
axins.axhline(v_th, color="gray", ls=":", lw=1.0)
axins.axvline(t_rel_ms, color="darkorange", ls="-.", lw=1.0, alpha=0.8)
axins.set_title("slip onset (detail)", fontsize=7)
axins.tick_params(labelsize=6); axins.grid(alpha=0.3)

fig3.tight_layout()
fig3.savefig("open_loop_slip.png", dpi=150)
print("saved open_loop_slip.png")
plt.close(fig3)