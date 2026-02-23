import json
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from functions.sor import sor
from functions.utils import compute_omega_opt, estimate_asymptotic_cr


# helpers
def safe_rmse(x_true, x_pred):
    x_true = np.asarray(x_true).ravel()
    x_pred = np.asarray(x_pred).ravel()
    if x_true.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean((x_true - x_pred) ** 2)))

def plot_iters_and_valley(omegas, iters, save_path, N, fontsize=20):
    valid = np.isfinite(iters) & (iters > 0)
    x, y = np.array(omegas)[valid], np.array(iters)[valid]
    if x.size == 0:
        return
    imin = int(np.argmin(y))
    L, R = max(0, imin-3), min(len(x)-1, imin+3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # left: full curve with highlighted valley region
    ax1.plot(x, y, linewidth=2.5)
    ax1.axvspan(x[L], x[R], color='gray', alpha=0.25)
    ax1.scatter([x[imin]], [y[imin]], marker='D', s=60, zorder=5, color='red')
    ax1.text(0.02, 0.02, f'ω={x[imin]:.3f}\niters={int(y[imin])}',
             transform=ax1.transAxes, fontsize=max(8, fontsize-4), color='red',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8), ha='left', va='bottom')
    ax1.set_xlabel('ω (omega)', fontsize=fontsize); ax1.set_ylabel('Number of iterations', fontsize=fontsize)
    ax1.set_title(f'SOR iterations for ω ∈ (0,2)', fontsize=fontsize); ax1.tick_params(labelsize=fontsize); ax1.grid(True)

    # right: zoomed valley
    ax2.plot(x[L:R+1], y[L:R+1], linewidth=2.5)
    ax2.scatter([x[imin]], [y[imin]], marker='D', s=60, zorder=5, color='red')
    ax2.text(0.02, 0.02, f'ω={x[imin]:.3f}\niters={int(y[imin])}',
             transform=ax2.transAxes, fontsize=max(8, fontsize-4), color='red',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8), ha='left', va='bottom')
    ax2.set_xlabel('ω (omega)', fontsize=fontsize); ax2.set_ylabel('Number of iterations', fontsize=fontsize)
    ax2.set_title(f'Gray Shaded region: ω∈[{x[L]:.3f},{x[R]:.3f}]', fontsize=fontsize)
    ax2.tick_params(labelsize=fontsize); ax2.grid(True)

    outp = os.path.join(save_path, f'iterations_vs_omega_N{N}_sidebyside.png')
    plt.tight_layout(); plt.savefig(outp, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'Saved: {outp}')

def run_baselines_sor(config, train_loader, max_iter=300):
    """
    Run SOR over a grid of omegas for each (A, x_true, b) in train_loader.
    Returns a dictionary with residual histories and a list of metric rows.
    Does NOT plot or save files.
    """
    print(f'Running SOR as a solver with different omega values in [0,2] ...')
    metrics_rows = []
    all_residuals_dict = {"baseline": {}, "SOR": {}, "plotted_lines": {}}
    optimal_omegas = {}
    
    for A, A_tensor, x_true, b in train_loader:
        N = A.shape[0]
        max_iter = N
        omega_opt_symmetric = compute_omega_opt(A)
        optimal_omegas[N] = omega_opt_symmetric
        config["default_omega"] = omega_opt_symmetric

        print(f'## N: {N}, max_iter: {max_iter}, opt_omega: {omega_opt_symmetric} ##')

        residuals_SOR_forN = {}
        if config.get("run_sor_as_solver", False):
            omegas = list(np.linspace(0.0, 2.0, num=config['n_actions'] + 2)[1:-1])
            for omega in omegas:
                
                try:
                    x0 = np.zeros_like(np.asarray(x_true).reshape(-1))
                    num_iters, x_SOR, resid_SOR = sor(A, b, x0, omega=omega,
                                                      tol=config["target_tol"], max_iter=max_iter)
                    resid_list = list(resid_SOR)
                    solution_error_SOR = safe_rmse(x_true, x_SOR)
                    final_resid = float(resid_list[-1]) if len(resid_list) > 0 else float('nan')
                    num_iters = int(num_iters)
                    convergence_rate = estimate_asymptotic_cr(resid_list)

                    residuals_SOR_forN[float(omega)] = resid_list
                    metrics_rows.append({
                        "matrix_size": int(N),
                        "method": "SOR",
                        "omega": float(omega),
                        "num_iterations": int(num_iters),
                        "final_solution_error": float(solution_error_SOR),
                        "final_residual": float(final_resid),
                        "convergence_rate": float(convergence_rate)
                    })
                except Exception as e:
                    residuals_SOR_forN[float(omega)] = []
                    metrics_rows.append({
                        "matrix_size": int(N),
                        "method": "SOR",
                        "omega": float(omega),
                        "num_iterations": 0,
                        "final_solution_error": float('nan'),
                        "final_residual": float('nan'),
                        "convergence_rate": float('nan')
                    })
                print(f'@@ omega: {omega:.6f}, num_iters: {num_iters}, solution_error_SOR: {solution_error_SOR}')
        else:
            # leave an empty dict for this N if SOR not run
            pass

        # ensure optimal omega is present (run it if missing)
        omega_opt = float(omega_opt_symmetric)
        if config.get("run_sor_as_solver", False):
            match_tol = 1e-6
            already_has_opt = any(np.isclose(o, omega_opt, atol=match_tol) for o in residuals_SOR_forN.keys())
            if not already_has_opt:
                try:
                    x0 = np.zeros_like(np.asarray(x_true).reshape(-1))
                    num_iters_opt, x_opt, resid_opt = sor(A, b, x0, omega=omega_opt,
                                                         tol=config["target_tol"], max_iter=max_iter)
                    resid_opt_list = list(resid_opt)
                    solution_error_opt = safe_rmse(x_true, x_opt)
                    final_resid_opt = float(resid_opt_list[-1]) if len(resid_opt_list) > 0 else float('nan')
                    num_iters_opt = int(num_iters_opt)
                    convergence_rate_opt = estimate_asymptotic_cr(resid_opt_list)

                    residuals_SOR_forN[float(omega_opt)] = resid_opt_list
                    metrics_rows.append({
                        "matrix_size": int(N),
                        "method": "SOR_opt",
                        "omega": float(omega_opt),
                        "num_iterations": int(num_iters_opt),
                        "final_solution_error": float(solution_error_opt),
                        "final_residual": float(final_resid_opt),
                        "convergence_rate": float(convergence_rate_opt)
                    })
                except Exception:
                    residuals_SOR_forN[float(omega_opt)] = []
            print(f'@@ optimal omega: {omega_opt}, num_iters: {num_iters_opt}, solution_error_SOR: {solution_error_opt}')

        all_residuals_dict["SOR"][int(N)] = residuals_SOR_forN
        # keep optimal omegas and empty plotted_lines placeholder
        all_residuals_dict["plotted_lines"][int(N)] = []

    return {
        "residuals_SOR": all_residuals_dict["SOR"],
        "plotted_lines": all_residuals_dict["plotted_lines"],
        "all_residuals": all_residuals_dict,
        "optimal_omegas": optimal_omegas,
        "metrics_rows": metrics_rows
    }

def plot_baseline_residuals_sor(all_residuals_dict, save_path="results/", fontsize=20):
    """
    Read all_residuals_dict and create per-matrix plots.
    - Every plotted line gets a color according to its omega (colorbar on right).
    - Legend contains only two entries:
        * optimal omega (thick blue dashed line)
        * RL omega (grid omega closest to optimal) (thick red solid line)
    - Adds a vertical colorbar on the right mapping omega in [0,2] to the colormap.
    """
    os.makedirs(save_path, exist_ok=True)

    sor_dict = all_residuals_dict.get("residuals_SOR", {})
    optimal_omegas = all_residuals_dict.get("optimal_omegas", {})
    metrics_rows = all_residuals_dict.get("metrics_rows", {})

    for N_key, residuals_SOR_forN in sorted(sor_dict.items(), key=lambda kv: int(kv[0])):
        if not residuals_SOR_forN:
            continue
        N = int(N_key)

        # Sort omegas numerically and collect items
        omegas_items = sorted(((float(k), list(v)) for k, v in residuals_SOR_forN.items()), key=lambda t: t[0])
        if len(omegas_items) == 0:
            continue

        # identify optimal omega and RL-closest omega
        omega_opt = float(optimal_omegas.get(int(N), np.nan))
        omega_grid = np.array([t[0] for t in omegas_items], dtype=float)
        if np.isnan(omega_opt):
            rl_omega = float(omega_grid[len(omega_grid)//2])
        else:
            idx_closest = int(np.argmin(np.abs(omega_grid - omega_opt)))
            rl_omega = float(omega_grid[idx_closest])

        # colormap and normalizer for omega range [0,2]
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=0.0, vmax=2.0)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # for colorbar

        # create figure & axes and leave room on right for colorbar and bottom for legend
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.88, bottom=0.26, left=0.08, right=0.82)

        opt_handle = None
        rl_handle = None

        # plot every line using color from omega
        for omega, resid in omegas_items:
            if not resid:
                continue
            color = cmap(norm(omega))
            x = range(1, len(resid) + 1)

            # make optimal omega a thick blue dashed line
            if (not np.isnan(omega_opt)) and np.isclose(omega, omega_opt, atol=1e-6):
                h, = ax.semilogy(x, resid, linewidth=3.5, linestyle='--', alpha=1.0,
                                 color='red', zorder=12)
                opt_handle = h
            # RL closest omega is thick red solid
            elif np.isclose(omega, rl_omega, atol=1e-3):
                h, = ax.semilogy(x, resid, linewidth=3.5, linestyle='-', alpha=1.0,
                                 color='blue', zorder=11)
                rl_handle = h
            else:
                # regular thin colored line
                h, = ax.semilogy(x, resid, linewidth=1.2, alpha=0.9, color=color, zorder=2)

        # labels, title, ticks
        ax.set_xlabel('Iteration', fontsize=fontsize, labelpad=12)
        ax.set_ylabel('Residual norm', fontsize=fontsize)
        ax.set_title(f'SOR Convergence for 2464 DOF', fontsize=fontsize)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.tick_params(axis='both', labelsize=fontsize)

        # legend with only the two highlighted handles
        legend_handles = []
        legend_labels = []
        if opt_handle is not None:
            legend_handles.append(opt_handle)
            legend_labels.append(f'Optimal ω = {omega_opt:.6f}')
        if (rl_handle is not None) and (rl_handle is not opt_handle):
            legend_handles.append(rl_handle)
            legend_labels.append(f'RL closest ω = {rl_omega:.6f}')

        if legend_handles:
            ncols = max(1, len(legend_handles))
            leg = ax.legend(legend_handles, legend_labels, loc='upper center',
                            bbox_to_anchor=(0.5, -0.18),
                            ncol=ncols,
                            fontsize=max(9, int(fontsize * 0.9)),
                            frameon=False,
                            handlelength=2.0,
                            handletextpad=0.6,
                            columnspacing=0.8)
            try:
                leg._legend_box.align = "center"
            except Exception:
                pass

        # colorbar on the right (vertical) showing omega scale 0->2
        cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # left, bottom, width, height in figure coords
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_label('ω', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize * 0.85)

        # save and close
        fig_filename = os.path.join(save_path, f'residuals_vs_omega_N{N}.png')
        plt.savefig(fig_filename, dpi=300, bbox_inches='tight', pad_inches=0.12)
        plt.close(fig)
        print(f'Saved the plot for SOR residuals (N={N}) to {fig_filename}')

        pdb.set_trace()

        rows_forN = [r for r in metrics_rows if int(r["matrix_size"]) == int(N) and r["method"].startswith("SOR")]
        rows_sorted = sorted(rows_forN, key=lambda r: r["omega"])
        omegas = np.array([r["omega"] for r in rows_sorted], dtype=float)
        iters = np.array([r["num_iterations"] for r in rows_sorted], dtype=float)
        plot_iters_and_valley(omegas, iters, save_path, N, fontsize=20)

        pdb.set_trace()

    return True