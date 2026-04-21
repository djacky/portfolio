"""
Test pyfresco H-infinity synthesis with coprime factorization for plants
that are unstable or contain integrators.  Specs for both plants:
    Ts = 5e-4 s,  desBw = 200 Hz,  desZ = 0.8,  desMm = 0.5,  order 6,
    n_integrators = 1.

Plants:
    (A) Inverted pendulum (linearized at upright, unstable)
            G(s) = (1/(m*L^2)) / (s^2 - g/L)
        unstable poles at s = +/- sqrt(g/L).
    (B) DC motor with integrator (position output)
            G(s) = Km / (s * (tau*s + 1))

Coprime factorization: choose F(s) = (s + a)^2 with a = 2*pi*desBw, so
    G(s) = N(s) / M(s)
    Pendulum:  N = 1/(m L^2) / (s+a)^2,     M = (s^2 - g/L) / (s+a)^2
    Motor:     N = Km / (s+a)^2,            M = s (tau s + 1) / (s+a)^2
Both N, M are stable, proper, and satisfy G = N/M on the jw-axis.

The SOCP substitution is straightforward: the original bisection constraint
    gamma |Wd (S + G*R - G*T)| <= Re(S + G*R)
becomes (multiply both sides by |M| and regroup via G = N/M)
    gamma |Wd (S*M + N*R - N*T)| <= Re(S*M + N*R)
with modulus-margin |mm*S*M| <= Re(S*M + N*R).  The closed-loop transfer
function T_cl = G*T/(S + G*R) is unchanged, so the post-hoc CL Bode uses
the original G.
"""

import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYFRESCO_ROOT = Path("E:/Projects/pyfresco")
PKG_ROOT = PYFRESCO_ROOT / "pyfresco" / "obcd"

pyfresco_pkg = types.ModuleType("pyfresco")
pyfresco_pkg.__path__ = [str(PYFRESCO_ROOT / "pyfresco")]
obcd_pkg = types.ModuleType("pyfresco.obcd")
obcd_pkg.__path__ = [str(PKG_ROOT)]
sys.modules["pyfresco"] = pyfresco_pkg
sys.modules["pyfresco.obcd"] = obcd_pkg


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        f"pyfresco.obcd.{name}", PKG_ROOT / filename
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"pyfresco.obcd.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


constants = load("constants", "constants.py")
# The SOCP bisection range has to be wide enough that some gamma is
# infeasible on the upper end; push g_max up for the easier cases.
constants.g_max = 1e4
common_funcs = load("common_funcs", "common_funcs.py")
solve_mod = load("solve", "solve.py")
OptAlgoIB_mod = load("OptAlgoIB", "OptAlgoIB.py")
OptAlgoIB = OptAlgoIB_mod.OptAlgoIB


class UiParams:
    def __init__(self):
        self.des_bw = 200.0
        self.des_z = 0.8
        self.des_mm = 0.5
        self.n_integrators = 1
        self.ref_delay = 0
        self.noise_rej = []


def print_cb(level, msg):
    safe = str(msg).encode("ascii", "replace").decode("ascii")
    print(f"[{level}] {safe}")


# ---------- global spec ------------------------------------------------
Ts = 5e-4
DES_BW = 200.0
A_POLE = 2 * np.pi * DES_BW   # coprime factorization pole (rad/s)
ORDER = {"n_r": 6, "n_s": 6, "n_t": 6}
w_nyq = np.pi / Ts            # ~6283 rad/s


# ---------- plant A: inverted pendulum (unstable) ----------------------
PEND_L = 0.5     # pendulum length [m]
PEND_M = 1.0     # tip mass [kg]
PEND_G = 9.81    # gravity [m/s^2]
PEND_WP2 = PEND_G / PEND_L           # 19.62 rad^2/s^2
PEND_KGAIN = 1.0 / (PEND_M * PEND_L * PEND_L)   # 4.0


def pendulum_G(w):
    # G(jw) = K / ((jw)^2 - wp^2) = -K / (w^2 + wp^2)
    denom = -(w * w) - PEND_WP2
    return PEND_KGAIN / denom + 0j


def pendulum_NM(w):
    # F(s) = (s+a)^2  =>  F(jw) = (a + jw)^2
    a = A_POLE
    F = (a + 1j * w) ** 2
    N = PEND_KGAIN / F
    A_poly = (1j * w) ** 2 - PEND_WP2   # s^2 - wp^2 at s=jw
    M = A_poly / F
    return N, M


# ---------- plant B: DC motor with integrator --------------------------
MOTOR_KM = 10.0
MOTOR_TAU = 0.05   # mechanical time constant [s]  => pole at -20 rad/s


def motor_G(w):
    # G(jw) = Km / (jw * (tau*jw + 1))
    jw = 1j * w
    denom = jw * (MOTOR_TAU * jw + 1.0)
    out = np.empty_like(w, dtype=complex)
    # Guard the grid point at w=0 if anyone ever includes it; we start > 0.
    mask = np.abs(denom) > 0
    out[mask] = MOTOR_KM / denom[mask]
    out[~mask] = 1e30
    return out


def motor_NM(w):
    a = A_POLE
    F = (a + 1j * w) ** 2
    jw = 1j * w
    N = MOTOR_KM / F
    A_poly = jw * (MOTOR_TAU * jw + 1.0)
    M = A_poly / F
    return N, M


# ---------- coprime-factorized rst_init --------------------------------
def rst_init_coprime(P, N_frf, M_frf, order, T_flag=False):
    """Phase-1 H-inf SOCP bisection with coprime factors N, M.

    The formulation mirrors OptAlgoIB.rst_init but substitutes
        G -> N            (stable numerator factor)
        S -> S * M        (baked into PSI)
    so that PSI = S*M + N*MA*R is a stable FRF even for unstable / type-1
    plants.  When the plant is stable we would pass N = G, M = 1 and
    recover the original formulation exactly.
    """
    Ro, So, To = OptAlgoIB.cont_struct(
        P, opt=True,
        n_r=order["n_r"], n_s=order["n_s"], n_t=order["n_t"],
    )

    gamma = cp.Parameter(nonneg=True, value=2 / 10)
    SM = cp.multiply(M_frf, So)
    PSI = SM + cp.multiply(N_frf * P.MA, Ro)

    F1 = (cp.multiply(gamma, cp.abs(cp.multiply(P.Wd,
            PSI - cp.multiply(N_frf, To)))) - cp.real(PSI))
    F2 = cp.abs(cp.multiply(P.user_pars.des_mm, SM)) - cp.real(PSI)

    if not T_flag:
        constraints = [F1 <= -1e-6, F2 <= -1e-6,
                       cp.real(P.So_no_int) >= 5e-3]
    else:
        constraints = [F1 <= -1e-6, F2 <= -1e-6,
                       cp.real(P.So_no_int) >= 5e-3,
                       cp.real(To) >= 5e-3]

    g_max, g_min = (constants.g_max, 0.1)
    bis = solve_mod.Solve(
        constraints, (P.rho_r, P.rho_s, P.rho_t), constants.bis_tol
    )
    bis_sol = bis.bisection(gamma, g_max, g_min, P.print_callback)

    if bis_sol["g-opt-bis"] > (1 / g_min) * 0.9:
        return {"gamma_opt": 100}

    R0, S0, T0 = OptAlgoIB.cont_struct(
        P, opt=False,
        r=bis_sol["x"][0], s=bis_sol["x"][1], t=bis_sol["x"][2],
    )
    GAIN = np.sum(bis_sol["x"][2]) / np.sum(bis_sol["x"][0])

    return {
        "gamma_opt": bis_sol["g-opt-bis"],
        "R_vec": bis_sol["x"][0],
        "S_vec": bis_sol["x"][1],
        "T_vec": bis_sol["x"][2],
        "Gain": GAIN,
        "Rf": R0, "Sf": S0, "Tf": T0,
    }


# ---------- closed-loop FRF (uses original G) --------------------------
def closed_loop_frf(w_eval, G_eval, R, S_full, T_full, Ts):
    def poly_z(coeffs, w):
        z_inv = np.exp(-1j * w * Ts)
        out = np.zeros_like(w, dtype=complex)
        for i, ci in enumerate(coeffs):
            out += ci * z_inv ** i
        return out

    Rz = poly_z(R, w_eval)
    Sz = poly_z(S_full, w_eval)
    Tz = poly_z(T_full, w_eval)
    return (G_eval * Tz) / (G_eval * Rz + Sz)


# ---------- generic driver ---------------------------------------------
def run_plant(label, G_fn, NM_fn, w_init, out_png, plot_title):
    print(f"\n========== {label}  (a_coprime = {A_POLE:.2f} rad/s) ==========")
    w = np.linspace(w_init, w_nyq, 200)
    G = G_fn(w)
    N_frf, M_frf = NM_fn(w)
    MA = np.ones_like(w, dtype=complex)
    up = UiParams()

    P = OptAlgoIB([G], MA, w, Ts, up, print_cb)
    init_out = rst_init_coprime(P, N_frf, M_frf, ORDER, T_flag=False)
    if init_out.get("gamma_opt", 0) == 100:
        print("rst_init_coprime INFEASIBLE")
        return None

    R = np.asarray(init_out["R_vec"])
    S_red = np.asarray(init_out["S_vec"])
    T = np.asarray(init_out["T_vec"])
    Gain = init_out["Gain"]
    gamma = init_out["gamma_opt"]

    s_base = np.concatenate(([1.0], S_red))
    S_full = np.convolve(s_base, [1.0, -1.0])   # bake the integrator
    T_norm = T / Gain

    print(f"[phase 1] gamma_opt = {gamma:.6f}")
    print(f"[phase 1] Gain      = {Gain:.6f}")
    print(f"[phase 1] R         = {np.array2string(R, precision=6, suppress_small=True)}")
    print(f"[phase 1] S_full    = {np.array2string(S_full, precision=6, suppress_small=True)}")
    print(f"[phase 1] T/Gain    = {np.array2string(T_norm, precision=6, suppress_small=True)}")

    # ---- plot ----
    w_plot = np.logspace(np.log10(w_init), np.log10(w_nyq), 1500)
    f_plot = w_plot / (2 * np.pi)
    G_plot = G_fn(w_plot)
    CL = closed_loop_frf(w_plot, G_plot, R, S_full, T_norm, Ts)

    mag_ol_db = 20 * np.log10(np.abs(G_plot) + 1e-30)
    phase_ol_deg = np.unwrap(np.angle(G_plot)) * 180 / np.pi
    mag_cl_db = 20 * np.log10(np.abs(CL) + 1e-30)
    phase_cl_deg = np.unwrap(np.angle(CL)) * 180 / np.pi

    fig, (ax_mag, ax_phase) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_mag.semilogx(f_plot, mag_ol_db, color="#6b7280", lw=1.2, ls="--",
                    label="open-loop |G|")
    ax_mag.semilogx(f_plot, mag_cl_db, color="#7c3aed", lw=1.6,
                    label="closed-loop |T_cl| (coprime rst_init)")
    ax_mag.axhline(-3, color="#9ca3af", lw=0.8, ls=":", label="-3 dB")
    ax_mag.axvline(DES_BW, color="#16a34a", lw=0.8, ls=":",
                   label=f"desired f_c = {DES_BW:.0f} Hz")
    ax_mag.axvline(A_POLE / (2 * np.pi), color="#f59e0b", lw=0.8, ls=":",
                   label=f"coprime pole a/(2pi) = {A_POLE/(2*np.pi):.0f} Hz")
    ax_mag.set_ylabel("magnitude [dB]")
    ax_mag.set_title(plot_title)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="lower left", fontsize=9)

    ax_phase.semilogx(f_plot, phase_ol_deg, color="#6b7280", lw=1.2, ls="--")
    ax_phase.semilogx(f_plot, phase_cl_deg, color="#7c3aed", lw=1.6)
    ax_phase.axvline(DES_BW, color="#16a34a", lw=0.8, ls=":")
    ax_phase.axvline(A_POLE / (2 * np.pi), color="#f59e0b", lw=0.8, ls=":")
    ax_phase.set_ylabel("phase [deg]")
    ax_phase.set_xlabel("frequency [Hz]")
    ax_phase.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
    return init_out


if __name__ == "__main__":
    here = Path(__file__).parent

    # Pendulum: unstable pole at sqrt(g/L) ~ 4.43 rad/s ~ 0.7 Hz.
    # Grid starts well below that so the synthesis sees the DC plateau.
    run_plant(
        label="inverted pendulum (unstable)",
        G_fn=pendulum_G,
        NM_fn=pendulum_NM,
        w_init=np.sqrt(PEND_WP2) / 100,
        out_png=here / "pyfresco-pendulum-closed-loop.png",
        plot_title=(
            "Coprime H-inf - inverted pendulum (unstable)\n"
            f"G = {PEND_KGAIN:.2f} / (s^2 - {PEND_WP2:.2f}),  "
            f"Ts={Ts*1e3:.2g} ms, des_bw={DES_BW:.0f} Hz, des_mm=0.5, des_z=0.8, order 6"
        ),
    )

    # Motor: dominant pole at 1/tau = 20 rad/s, plus an integrator.
    run_plant(
        label="DC motor (integrator)",
        G_fn=motor_G,
        NM_fn=motor_NM,
        w_init=(1.0 / MOTOR_TAU) / 100,
        out_png=here / "pyfresco-motor-closed-loop.png",
        plot_title=(
            "Coprime H-inf - DC motor with integrator\n"
            f"G = {MOTOR_KM:.0f} / (s (tau s + 1)),  tau={MOTOR_TAU}s,  "
            f"Ts={Ts*1e3:.2g} ms, des_bw={DES_BW:.0f} Hz, des_mm=0.5, des_z=0.8, order 6"
        ),
    )
