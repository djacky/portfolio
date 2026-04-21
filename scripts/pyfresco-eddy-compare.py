"""
Compare browser H∞ rst_init output against pyfresco for Plant 3
(eddy-current magnet, fractional α=0.75 · −15 dB/dec).

Plant:   G(s) = (1 / (1 + (s·τ)^α)) · e^{-s·τ_d}
         K = 1,  τ = 1/(2π·20),  α = 0.75,  τ_d = 120 μs
Specs:   Ts = 1e-4,  desBw = Fs/15 ≈ 666.67 Hz,  desMm = 0.5, desZ = 0.8
Order:   n_r = n_s = n_t = 6,  n_integrators = 1
"""

import sys, types, importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYFRESCO_ROOT = Path("E:/Projects/pyfresco")
PKG_ROOT = PYFRESCO_ROOT / "pyfresco" / "obcd"

# Shim pyfresco.obcd package (avoids the CERN-only `pyfgc` import).
pyfresco_pkg = types.ModuleType("pyfresco"); pyfresco_pkg.__path__ = [str(PYFRESCO_ROOT / "pyfresco")]
obcd_pkg = types.ModuleType("pyfresco.obcd"); obcd_pkg.__path__ = [str(PKG_ROOT)]
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
# Push g_max up so the bisection can bracket (same trick as the dipole script).
constants.g_max = 1e4
common_funcs = load("common_funcs", "common_funcs.py")
solve_mod = load("solve", "solve.py")
OptAlgoIB_mod = load("OptAlgoIB", "OptAlgoIB.py")
OptAlgoIB = OptAlgoIB_mod.OptAlgoIB

# ---- pyfresco hinf() bug workaround ----------------------------------
# Upstream hinf's early-exit branch `elif (g_lmi[it] < 1 and it > 2):
# return {'gamma_opt': g_lmi[it]}` drops the controller coefficients
# when γ converges below 1.  We patch the method to always return the
# last feasible R/S/T.  See pyfresco-compare.py for the full rationale.
import cvxpy as _cp
import numpy as _np


def _hinf_patched(self, R0, S0, order, T_flag):
    Ro, So, To = OptAlgoIB.cont_struct(
        self, opt=True, n_r=order['n_r'], n_s=order['n_s'], n_t=order['n_t'])

    g_lmi = _np.insert(_np.zeros(100), 0, 1e6)
    it = 1
    last_bis = None
    last_R0 = R0
    last_S0 = S0
    last_T0 = None
    last_GAIN = 1.0

    while g_lmi[it - 1] - g_lmi[it] > constants.LMI_tol:
        gamma = _cp.Parameter(nonneg=True, value=2 / 200)
        constraints = []
        for G in self.G_multi:
            PSI2 = So + _cp.multiply(G * self.MA, Ro)
            PSI2_0 = last_S0 + _np.multiply(G * self.MA, last_R0)

            t1 = (2 * _cp.real(_cp.multiply(_cp.conj(PSI2_0), PSI2))
                  - _cp.power(_cp.abs(PSI2_0), 2))
            x1 = _cp.multiply(
                gamma,
                _cp.power(_cp.abs(_cp.multiply(self.Wd, PSI2 - _cp.multiply(G, To))), 2))
            x2 = _cp.power(_cp.abs(_cp.multiply(self.user_pars.des_mm, So)), 2)

            con_noise = []
            if self.user_pars.noise_rej:
                for pair in self.user_pars.noise_rej:
                    index = _np.where(self.w == 2 * _np.pi * pair[0])[0][0]
                    gain_abs = 10 ** (-pair[1] / 20)
                    xn = _cp.power(
                        _cp.abs(_cp.multiply(gain_abs * G[index], So[index])), 2)
                    con_noise += [xn - t1[index] <= -1e-6]

            g_max, g_min = (10, 1e-3)
            if not T_flag:
                constraints += [x1 - t1 <= -1e-6, x2 - t1 <= -1e-6,
                                _cp.real(self.So_no_int) >= 5e-3] + con_noise
            else:
                constraints += [x1 - t1 <= -1e-6, x2 - t1 <= -1e-6,
                                _cp.real(self.So_no_int) >= 5e-3,
                                _cp.real(To) >= 5e-3] + con_noise

        bis = solve_mod.Solve(
            constraints, (self.rho_r, self.rho_s, self.rho_t), constants.bis_tol)
        bis_sol = bis.bisection(gamma, g_max, g_min, self.print_callback)
        last_bis = bis_sol

        g_lmi[1] = 1e6
        g_lmi[it + 1] = _np.sqrt(bis_sol['g-opt-bis'])
        it = it + 1

        last_GAIN = _np.sum(bis_sol['x'][2]) / _np.sum(bis_sol['x'][0])
        last_R0, last_S0, last_T0 = OptAlgoIB.cont_struct(
            self, opt=False,
            r=bis_sol['x'][0], s=bis_sol['x'][1], t=bis_sol['x'][2])

        if bis_sol['g-opt-bis'] > constants.g_thresh:
            break
        elif (g_lmi[it] < 1 and it > 2):
            break
        elif (g_lmi[it] > g_lmi[it - 1] and it > 2):
            break
        else:
            self.print_callback(
                'info',
                f'Feasible for iteration {it - 1} (gamma = {round(g_lmi[it], constants.g_digits)})')

    T0_norm = last_T0 / last_GAIN
    return {'gamma_opt': _np.sqrt(last_bis['g-opt-bis']),
            'R_vec': last_bis['x'][0],
            'S_vec': last_bis['x'][1],
            'T_vec': last_bis['x'][2],
            'Gain': last_GAIN,
            'Rf': last_R0, 'Sf': last_S0, 'Tf': T0_norm}


OptAlgoIB.hinf = _hinf_patched


class UiParams:
    def __init__(self):
        self.des_bw = 10000.0 / 15      # Fs / 15 (Plant 3 default)
        self.des_z = 0.8
        self.des_mm = 0.5
        self.n_integrators = 1
        self.ref_delay = 0
        self.noise_rej = []


def print_cb(level, msg):
    # pyfresco embeds Unicode (γ, ω, …) which cp1252 stdout can't encode.
    safe = str(msg).encode("ascii", "replace").decode("ascii")
    print(f"[{level}] {safe}")


# ---------- plant: eddy-current magnet (fractional α=0.75) -------------
TAU, ALPHA, TAU_D = 1.0 / (2 * np.pi * 20), 0.75, 120e-6
Ts = 1e-4
w_nyq = np.pi / Ts                # ~31416 rad/s
w_init = (2 * np.pi * 20) / 100   # dominant-pole / 100 ≈ 1.257 rad/s


def plant_frf(w):
    # (j ω τ)^α  via polar form  (handles non-integer α).
    jWt = 1j * w * TAU
    pow_val = np.exp(ALPHA * np.log(jWt))      # complex power
    return np.exp(-1j * w * TAU_D) / (1.0 + pow_val)


# Grid A: pyfresco log grid (~282 pts for order-6).
nu = 3 * 6
eps, beta = constants.epsilon, constants.beta
Nw = int(np.ceil((1 / eps) * (np.log(1 / beta) + nu - 1
                              + np.sqrt(2 * (nu - 1) * np.log(1 / beta)))))
w_log = np.logspace(np.log10(w_init), np.log10(w_nyq), Nw)
# Grid B: browser linear 200-pt grid.
w_lin = np.linspace(w_init, w_nyq, 200)


def closed_loop_frf(w_eval, R, S_full, T_full, Ts):
    G = plant_frf(w_eval)
    def poly_z(coeffs, w):
        z_inv = np.exp(-1j * w * Ts)
        out = np.zeros_like(w, dtype=complex)
        for i, c in enumerate(coeffs):
            out += c * z_inv ** i
        return out
    Rz = poly_z(R, w_eval)
    Sz = poly_z(S_full, w_eval)
    Tz = poly_z(T_full, w_eval)
    return (G * Tz) / (G * Rz + Sz)


def _unpack(out):
    R = np.asarray(out["R_vec"])
    S_red = np.asarray(out["S_vec"])
    T = np.asarray(out["T_vec"])
    Gain = out["Gain"]
    s_base = np.concatenate(([1.0], S_red))
    S_full = np.convolve(s_base, [1.0, -1.0])
    T_norm = T / Gain
    return R, S_full, T_norm, Gain


def run(w, label):
    print(f"\n========== {label} (Nw={len(w)}) ==========")
    G = plant_frf(w)
    MA = np.ones_like(w, dtype=complex)
    up = UiParams()
    P = OptAlgoIB([G], MA, w, Ts, up, print_cb)

    # ---------- Phase 1: rst_init (SOCP bisection) ----------
    init_out = P.rst_init({"n_r": 6, "n_s": 6, "n_t": 6}, T_flag=False)
    if init_out.get("gamma_opt", 0) == 100:
        print("rst_init INFEASIBLE")
        return None, None
    R, S_full, T_norm, Gain = _unpack(init_out)
    gamma = init_out["gamma_opt"]
    print(f"[phase 1] gamma_opt = {gamma:.6f}")
    print(f"[phase 1] Gain      = {Gain:.6f}")
    print(f"[phase 1] R         = {np.array2string(R, precision=6, suppress_small=True)}")
    print(f"[phase 1] S_full    = {np.array2string(S_full, precision=6, suppress_small=True)}")
    print(f"[phase 1] T/Gain    = {np.array2string(T_norm, precision=6, suppress_small=True)}")
    init_sol = {"R": R, "S_full": S_full, "T": T_norm, "gamma": gamma}

    # ---------- Phase 2: hinf (LMI-linearized refinement) ----------
    print(f"\n---------- Phase 2: hinf LMI refinement ----------")
    hinf_out = P.hinf(init_out["Rf"], init_out["Sf"],
                      {"n_r": 6, "n_s": 6, "n_t": 6}, T_flag=False)
    gv = hinf_out.get("gamma_opt", 0)
    if gv in (100, 1000):
        print("hinf INFEASIBLE")
        return init_sol, None
    R2, S_full2, T_norm2, Gain2 = _unpack(hinf_out)
    gamma2 = hinf_out["gamma_opt"]
    print(f"[phase 2] gamma_opt = {gamma2:.6f}")
    print(f"[phase 2] Gain      = {Gain2:.6f}")
    print(f"[phase 2] R         = {np.array2string(R2, precision=6, suppress_small=True)}")
    print(f"[phase 2] S_full    = {np.array2string(S_full2, precision=6, suppress_small=True)}")
    print(f"[phase 2] T/Gain    = {np.array2string(T_norm2, precision=6, suppress_small=True)}")
    hinf_sol = {"R": R2, "S_full": S_full2, "T": T_norm2, "gamma": gamma2}
    return init_sol, hinf_sol


def plot_closed_loops(log_sol, lin_sol, log_hinf_sol, browser_sol, out_path):
    w_plot = np.logspace(np.log10(w_init), np.log10(w_nyq), 1500)
    f_plot = w_plot / (2 * np.pi)

    fig, (ax_mag, ax_phase) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    for sol, color, name in [
        (log_sol,      "#2563eb", "pyfresco rst_init log (282 pts)"),
        (lin_sol,      "#dc2626", "pyfresco rst_init linear (200 pts)"),
        (log_hinf_sol, "#7c3aed", "pyfresco rst_init + hinf (refined)"),
        (browser_sol,  "#16a34a", "browser TS (WD_CAP removed)"),
    ]:
        if sol is None:
            continue
        CL = closed_loop_frf(w_plot, sol["R"], sol["S_full"], sol["T"], Ts)
        mag_db = 20 * np.log10(np.abs(CL) + 1e-30)
        phase_deg = np.unwrap(np.angle(CL)) * 180 / np.pi
        ax_mag.semilogx(f_plot, mag_db, color=color, lw=1.4, label=name)
        ax_phase.semilogx(f_plot, phase_deg, color=color, lw=1.4)

    ax_mag.axhline(-3, color="#6b7280", lw=0.8, ls="--", label="-3 dB")
    ax_mag.axvline(UiParams().des_bw, color="#16a34a", lw=0.8, ls=":",
                   label=f"desired f_c = {UiParams().des_bw:.0f} Hz")
    ax_mag.set_ylabel("|T_cl|  [dB]")
    ax_mag.set_title(
        "Closed-loop Bode — eddy-current magnet (fractional α=0.75)\n"
        f"Ts={Ts*1e3:.2g} ms, des_bw={UiParams().des_bw:.1f} Hz, "
        "des_mm=0.5, des_z=0.8, order 6"
    )
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(loc="lower left", fontsize=9)

    ax_phase.axvline(UiParams().des_bw, color="#16a34a", lw=0.8, ls=":")
    ax_phase.set_ylabel("phase  [deg]")
    ax_phase.set_xlabel("frequency  [Hz]")
    ax_phase.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    log_sol, log_hinf_sol = run(w_log, "pyfresco default (logspace)")
    lin_sol, _            = run(w_lin, "browser-style (linspace, 200 pts)")

    # Browser TS synthesis output captured via
    # `npx tsx scripts/hinf-eddy-usercase.ts` (Ts=1e-4, desBw=666.67 Hz,
    # desMm=0.5, desZ=0.8).  H∞ = 1.102, f_c = 672.9 Hz.
    browser_sol = {
        "R": np.array([19.510202, -6.099045, -5.585237, -12.832245, 3.666978, 3.647486]),
        "S_full": np.array([1.000000, -0.392279, -0.671952, -0.329292, 0.056356, 0.337167]),
        "T": np.array([7.221023, -0.508315, 0.425771, -2.656328, -1.102063, -1.071948]),
    }

    out = Path(__file__).parent / "pyfresco-eddy-closed-loop.png"
    plot_closed_loops(log_sol, lin_sol, log_hinf_sol, browser_sol, out)
