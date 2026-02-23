# Standard library
import os
import pdb
import json

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Local
from functions.fgmres import FlexibleGMRES_original
from functions.preconditioners import M_sor
from functions.utils import (
    compute_omega_opt,
    estimate_asymptotic_cr
)


# helpers
def safe_rmse(x_true, x_pred):
    x_true = np.asarray(x_true).ravel()
    x_pred = np.asarray(x_pred).ravel()
    if x_true.size == 0:
        return float('nan')
    return float(np.sqrt(np.mean((x_true - x_pred) ** 2)))

def plot_iters_and_valley(omegas, iters, save_path, N, index, fontsize=20):
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
    ax1.set_title(f'FGMRES with $M_{{SOR}}(\\omega)$ for {N} DOF', fontsize=fontsize); ax1.tick_params(labelsize=fontsize); ax1.grid(True)

    # right: zoomed valley
    ax2.plot(x[L:R+1], y[L:R+1], linewidth=2.5)
    ax2.scatter([x[imin]], [y[imin]], marker='D', s=60, zorder=5, color='red')
    ax2.text(0.02, 0.02, f'ω={x[imin]:.3f}\niters={int(y[imin])}',
             transform=ax2.transAxes, fontsize=max(8, fontsize-4), color='red',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8), ha='left', va='bottom')
    ax2.set_xlabel('ω (omega)', fontsize=fontsize); ax2.set_ylabel('Number of iterations', fontsize=fontsize)
    ax2.set_title(f'Shaded region — ω∈[{x[L]:.3f},{x[R]:.3f}]', fontsize=fontsize)
    ax2.tick_params(labelsize=fontsize); ax2.grid(True)

    outp = os.path.join(save_path, f'{index}_iterations_vs_omega_N{N}_sidebyside_fgmres.png')
    plt.tight_layout(); plt.savefig(outp, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'Saved: {outp}')


def run_baselines_fgmres_and_plot(config, train_loader, save_path):
    """
    Runs baseline (no preconditioner) and SOR-preconditioned FGMRES
    across a grid of omega in (0,2). Produces:
      - one semilogy figure per matrix showing all residual histories (baseline + omegas)
      - an Excel file with metrics (num_iterations, final_solution_error, final_residual, convergence_rate)
      - JSON file(s) with residual histories
    """

    print('\n Running baselines ..... \n')

    # containers for table and residuals
    metrics_rows = []
    all_residuals_dict = {"baseline": {}, "SOR": {}, "plotted_lines": {}}

    index = 0
    optimal_omegas = {}
    for A, A_tensor, x_true, b in train_loader:
        N = A.shape[0]
        print(f'\n------------------ Matrix size N = {N} --------------------')

        # compute per-matrix diagnostics
        if config["dataset"]=='diffusion':
            omega_opt_symmetric = compute_omega_opt(A)
            optimal_omegas[N] = omega_opt_symmetric
            config["default_omega"] = omega_opt_symmetric

        # ----------------------------------------------------------------
        # Baseline: Flexible GMRES with no preconditioner
        # ----------------------------------------------------------------
        residuals_baseline_forN = []
        convergence_rate_baseline = float('nan')
        if config.get("run_FGMRES_baseline", False):
            print('\n Running FGMRES baseline (no preconditioner) ... \n')
            try:
                gmres_baseline = FlexibleGMRES_original(A, max_iter=config["max_iter"], tol=config["target_tol"])
                x_baseline, resid_baseline = gmres_baseline.solve(b)
                residuals_baseline_forN = list(resid_baseline)
                solution_error_baseline = safe_rmse(x_true, x_baseline)
                final_resid_baseline = float(residuals_baseline_forN[-1]) if len(residuals_baseline_forN) > 0 else float('nan')
                num_iters_baseline = len(residuals_baseline_forN)
                # compute convergence rate for baseline
                convergence_rate_baseline = estimate_asymptotic_cr(residuals_baseline_forN)
                print(f' baseline: iters={num_iters_baseline}, final_resid={final_resid_baseline:.3e}, sol_err={solution_error_baseline:.3e}, cr={convergence_rate_baseline:.6f}')
            except Exception as e:
                print(f' Baseline solve FAILED for N={N}: {e}')
                residuals_baseline_forN = []
                solution_error_baseline = float('nan')
                final_resid_baseline = float('nan')
                num_iters_baseline = 0
                convergence_rate_baseline = float('nan')

            # record baseline row in metrics (now includes convergence_rate)
            metrics_rows.append({
                "matrix_size": int(N),
                "method": "baseline",
                "omega": "baseline",
                "num_iterations": int(num_iters_baseline),
                "final_solution_error": float(solution_error_baseline),
                "final_residual": float(final_resid_baseline),
                "convergence_rate": float(convergence_rate_baseline)
            })
        else:
            print('Skipping baseline (config run_FGMRES_baseline is False).')

        # store baseline residuals
        all_residuals_dict["baseline"][int(N)] = residuals_baseline_forN

        # ----------------------------------------------------------------
        # SOR: loop over omegas
        # ----------------------------------------------------------------
        residuals_SOR_forN = {}
        if config.get("run_FGMRES_default_SOR", False):
            print('\n Running FGMRES + SOR over grid of omega values ... \n')
            if config["dataset"]=="diffusion":
                omegas = list(np.linspace(0.0, 2.0, num=config['n_actions'] + 2)[1:-1])  # excludes 0 and 2 endpoints
            elif config["dataset"]=="advection":
                omegas = list(np.linspace(config["omega_min"], config["omega_max"], num=config['n_actions'] + 2)[1:-1])  # excludes 0 and 2 endpoints

            for omega in omegas:
                try:
                    M_sor_ = M_sor(A, omega=omega)
                    gmres_SOR = FlexibleGMRES_original(A,
                                                       max_iter=config["max_iter"],
                                                       tol=config["target_tol"],
                                                       M=M_sor_,
                                                       omega=omega)
                    x_SOR, resid_SOR = gmres_SOR.solve(b)
                    resid_list = list(resid_SOR)
                    solution_error_SOR = safe_rmse(x_true, x_SOR)
                    final_resid = float(resid_list[-1]) if len(resid_list) > 0 else float('nan')
                    num_iters = len(resid_list)
                    # compute convergence rate for this omega
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

                    print(f' ω={omega:.3f}  iters={num_iters:3d}  final_resid={final_resid:.3e}  sol_err={solution_error_SOR:.3e}  cr={convergence_rate:.6f}')

                except Exception as e:
                    print(f' ω={omega:.3f}  FAILED: {e}')
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
        else:
            print('Skipping SOR sweep (config run_FGMRES_default_SOR is False).')

        # store SOR residuals for this matrix
        all_residuals_dict["SOR"][int(N)] = residuals_SOR_forN

        # ----------------------------------------------------------------
        # If optimal omega is not exactly in the sweep, run it explicitly so we have its residuals & cr
        # ----------------------------------------------------------------
        # Note: we will still keep the compute_omega_opt for diagnostics, but we will
        # define the "optimal" omega for plotting/labeling as the omega with minimum iterations.
        if config["dataset"]=='diffusion':
            # keep the theoretical omega value from compute_omega_opt if available
            omega_opt_theoretical = float(omega_opt_symmetric)

            # tolerance for matching opt to grid element
            match_tol = 1e-6
            if config.get("run_FGMRES_default_SOR", False):
                already_has_opt = any(np.isclose(o, omega_opt_theoretical, atol=match_tol) for o in residuals_SOR_forN.keys())
                if not already_has_opt:
                    # run theoretical optimal omega explicitly to have its residuals (if desired)
                    try:
                        M_sor_opt = M_sor(A, omega=omega_opt_theoretical)
                        gmres_opt = FlexibleGMRES_original(A, max_iter=config["max_iter"], tol=config["target_tol"], M=M_sor_opt, omega=omega_opt_theoretical)
                        x_opt, resid_opt = gmres_opt.solve(b)
                        resid_opt_list = list(resid_opt)
                        solution_error_opt = safe_rmse(x_true, x_opt)
                        final_resid_opt = float(resid_opt_list[-1]) if len(resid_opt_list) > 0 else float('nan')
                        num_iters_opt = len(resid_opt_list)
                        convergence_rate_opt = estimate_asymptotic_cr(resid_opt_list)

                        # store
                        residuals_SOR_forN[float(omega_opt_theoretical)] = resid_opt_list
                        metrics_rows.append({
                            "matrix_size": int(N),
                            "method": "SOR_opt",
                            "omega": float(omega_opt_theoretical),
                            "num_iterations": int(num_iters_opt),
                            "final_solution_error": float(solution_error_opt),
                            "final_residual": float(final_resid_opt),
                            "convergence_rate": float(convergence_rate_opt)
                        })
                        print(f' OPT ω={omega_opt_theoretical:.6f}  iters={num_iters_opt:3d}  final_resid={final_resid_opt:.3e}  sol_err={solution_error_opt:.3e}  cr={convergence_rate_opt:.6f}')
                    except Exception as e:
                        print(f' OPT ω={omega_opt_theoretical:.6f} FAILED: {e}')
                        residuals_SOR_forN[float(omega_opt_theoretical)] = []
                        metrics_rows.append({
                            "matrix_size": int(N),
                            "method": "SOR_opt",
                            "omega": float(omega_opt_theoretical),
                            "num_iterations": 0,
                            "final_solution_error": float('nan'),
                            "final_residual": float('nan'),
                            "convergence_rate": float('nan')
                        })

        # ----------------------------------------------------------------
        # Determine the "optimal omega" for labeling/highlighting: choose the omega with minimum num_iterations
        # ----------------------------------------------------------------
        omega_opt_val = np.nan
        # extract SOR rows for this N
        rows_forN = [r for r in metrics_rows if int(r["matrix_size"]) == int(N) and r["method"].startswith("SOR")]
        if len(rows_forN) > 0:
            # ignore entries with zero iterations (failed runs)
            valid_rows = [r for r in rows_forN if r["num_iterations"] > 0]
            if len(valid_rows) > 0:
                # pick the omega with the minimum number of iterations (if tie, pick first)
                best_row = min(valid_rows, key=lambda r: r["num_iterations"])
                omega_opt_val = float(best_row["omega"])
            else:
                omega_opt_val = np.nan
        else:
            omega_opt_val = np.nan

        # ----------------------------------------------------------------
        # Plot: one figure per matrix size, all lines (baseline + omegas) on same axes
        # ----------------------------------------------------------------

        # --- START: replaced plotting block: match plot_baseline_residuals_sor style ---
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        # Prepare items
        sorted_items = sorted(((float(k), list(v)) for k, v in residuals_SOR_forN.items()), key=lambda t: t[0])
        if len(sorted_items) == 0:
            # nothing to plot: still create an empty saved figure with baseline if present
            fig, ax = plt.subplots(figsize=(10, 6))
            if residuals_baseline_forN:
                ax.semilogy(range(1, len(residuals_baseline_forN) + 1), residuals_baseline_forN,
                            label='baseline (no precond)', linewidth=3.0, color='k')
            ax.set_xlabel('Iteration', fontsize=20, labelpad=12)
            ax.set_ylabel('Residual norm', fontsize=20)
            # ax.set_title(f'SOR Convergence for {N} DOF', fontsize=20)
            ax.set_title(f'FGMRES Convergence with $M_{{SOR}}(\\omega)$ for {N} DOF', fontsize=20)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
            fig_filename = os.path.join(save_path, f'residuals_vs_omega_N{N}.png')
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight', pad_inches=0.12)
            plt.close(fig)
            print(f'Saved residual plot to: {fig_filename}')
        else:
            omegas_items = sorted_items
            # Use omega_opt_val determined from metric (min num_iterations). If nan, fallback to middle of grid for plotting behavior
            if np.isnan(omega_opt_val):
                omega_grid = np.array([t[0] for t in omegas_items], dtype=float)
                omega_opt_val = float(omega_grid[len(omega_grid)//2]) if len(omega_grid) > 0 else np.nan

            cmap = plt.get_cmap("viridis")
            norm = Normalize(vmin=0.0, vmax=2.0)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            fig.subplots_adjust(top=0.88, bottom=0.26, left=0.08, right=0.82)

            opt_handle = None

            # plot every line using color from omega
            for omega, resid in omegas_items:
                if not resid:
                    continue
                color = cmap(norm(omega))
                x = range(1, len(resid) + 1)

                # make optimal omega (based on minimum iterations) a thick red dashed line (highlight)
                if (not np.isnan(omega_opt_val)) and np.isclose(omega, omega_opt_val, atol=1e-6):
                    h, = ax.semilogy(x, resid, linewidth=3.5, linestyle='--', alpha=1.0,
                                     color='red', zorder=12)
                    opt_handle = h
                else:
                    # regular thin colored line
                    h, = ax.semilogy(x, resid, linewidth=1.2, alpha=0.9, color=color, zorder=2)

            # also plot baseline prominently if available
            if residuals_baseline_forN:
                ax.semilogy(range(1, len(residuals_baseline_forN) + 1),
                            residuals_baseline_forN,
                            label='baseline (no precond)',
                            linewidth=3.0,
                            color='k')

            # labels, title, ticks
            fontsize = 20
            ax.set_xlabel('Iteration', fontsize=fontsize, labelpad=12)
            ax.set_ylabel('Residual norm', fontsize=fontsize)
            ax.set_title(f'FGMRES Convergence with $M_{{SOR}}(\\omega)$ for {N} DOF', fontsize=fontsize)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
            ax.tick_params(axis='both', labelsize=fontsize)

            # legend with only the highlighted theoretical/selected handle if present
            legend_handles = []
            legend_labels = []
            if opt_handle is not None:
                legend_handles.append(opt_handle)
                legend_labels.append(f'Optimal ω (min iters) = {omega_opt_val:.6f}')

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

            # colorbar on the right (vertical) showing omega scale 0..2
            cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # left, bottom, width, height in figure coords
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
            cbar.set_label('ω', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize * 0.85)

            # save and close
            fig_filename = os.path.join(save_path, f'{index}_residuals_vs_omega_N{N}_fgmres.png')
            plt.savefig(fig_filename, dpi=300, bbox_inches='tight', pad_inches=0.12)
            plt.close(fig)
            print(f'Saved residual plot to: {fig_filename}')
        # --- END: replaced plotting block ---

        # save the plotted-lines metadata (residuals + styles) for this matrix size
        # (keep the old metadata collection to avoid breaking downstream code)
        plotted_lines_forN = []
        if residuals_baseline_forN:
            plotted_lines_forN.append({
                "label": 'baseline (no precond)',
                "omega": "baseline",
                "residuals": list(residuals_baseline_forN),
                "color": 'k',
                "linewidth": 3.0,
                "linestyle": '-',
                "is_opt": False
            })

        for omega, resid in sorted(((float(k), v) for k, v in residuals_SOR_forN.items()), key=lambda t: t[0]):
            if not resid:
                continue
            # mark as optimal if it's close to omega_opt_val (min iterations)
            is_opt = (not np.isnan(omega_opt_val)) and np.isclose(omega, float(omega_opt_val), atol=1e-3)
            cr = estimate_asymptotic_cr(resid)
            stored_label = f'ω={omega:.3f} (n={len(resid)}, cr={cr:.3f})' if (is_opt) else f'ω={omega:.3f} (unlabeled)'
            plotted_lines_forN.append({
                "label": stored_label,
                "omega": float(omega),
                "residuals": list(resid),
                "color": cmap(norm(omega)),
                "linewidth": 3.5 if is_opt else 1.2,
                "linestyle": '--' if is_opt else '-',
                "is_opt": bool(is_opt)
            })

        all_residuals_dict["plotted_lines"][int(N)] = plotted_lines_forN

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        optimal_omegas[N] = omega_opt_val

        rows_forN = [r for r in metrics_rows if int(r["matrix_size"]) == int(N) and r["method"].startswith("SOR")]
        rows_sorted = sorted(rows_forN, key=lambda r: r["omega"])
        omegas = np.array([r["omega"] for r in rows_sorted], dtype=float)
        iters = np.array([r["num_iterations"] for r in rows_sorted], dtype=float)
        plot_iters_and_valley(omegas, iters, save_path, N, index, fontsize=20)

        index += 1

    # ----------------------------------------------------------------
    # After processing all matrices: build DataFrame and save to Excel + JSON
    # ----------------------------------------------------------------
    if len(metrics_rows) == 0:
        print('No metrics were collected (check run_FGMRES_baseline/run_FGMRES_default_SOR flags).')
    else:
        df_metrics = pd.DataFrame(metrics_rows,
                                  columns=["matrix_size", "method", "omega", "num_iterations", "final_solution_error", "final_residual", "convergence_rate"])
        excel_path = os.path.join(save_path, 'metrics_omega_comparison.xlsx')
        df_metrics.to_excel(excel_path, index=False)
        print(f'Saved metrics table to: {excel_path}')

    # Save residual histories (JSON) - convert numpy types / arrays to lists
    json_path = os.path.join(save_path, 'all_residuals.json')
    with open(json_path, 'w') as f:
        json.dump(all_residuals_dict, f, indent=2, default=lambda x: list(x) if isinstance(x, (np.ndarray,)) else x)
    print(f'Saved residual histories to: {json_path}')

    # Print a compact summary top rows
    if len(metrics_rows) > 0:
        print('\nMetrics summary (first 12 rows):')
        print(pd.DataFrame(metrics_rows).head(12).to_string(index=False))

    # Return residuals and metrics to the caller
    return {
        "residuals_baseline": all_residuals_dict["baseline"],
        "residuals_SOR": all_residuals_dict["SOR"],
        "plotted_lines": all_residuals_dict["plotted_lines"],
        "all_residuals": all_residuals_dict,
        "optimal_omegas": optimal_omegas,
        "metrics_df": df_metrics if 'df_metrics' in locals() else None
    }

