import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from scipy.sparse import csr_matrix, diags, eye, identity, issparse
from scipy.sparse.linalg import eigs, LinearOperator


fontsize=20

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def matrix_to_graph(A):
    coo = A.tocoo()
    row, col, data = coo.row, coo.col, coo.data
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr  = torch.tensor(data, dtype=torch.float).unsqueeze(1)  # shape (E,1)
    node_attr = torch.tensor(np.asarray(A.sum(axis=1)).ravel(), dtype=torch.float).unsqueeze(1)  # shape (E,1)
    return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

def is_consistently_ordered(A):
    """Return True iff A's adjacency graph admits the consistent ordering (Def.4.13)."""
    A = csr_matrix(A)                         # accept dense or sparse
    pattern = (A != 0).astype(int).tocsr()    # ensure CSR
    P = pattern + pattern.T.tocsr()           # symmetric pattern in CSR
    P.setdiag(0); P.eliminate_zeros()

    indptr, indices = P.indptr, P.indices
    n = P.shape[0]
    SENT = np.iinfo(np.int32).min
    layer = np.full(n, SENT, dtype=np.int32)
    for s in range(n):
        if layer[s] != SENT: continue
        layer[s] = 0
        stack = [s]
        while stack:
            i = stack.pop()
            for k in range(indptr[i], indptr[i+1]):
                j = indices[k]
                if j == i: continue
                expected = layer[i] + (1 if j > i else -1)
                if layer[j] == SENT:
                    layer[j] = expected
                    stack.append(j)
                elif layer[j] != expected:
                    return False
    return True

def rmse(x_true, x_pred):
    x_true = np.asarray(x_true).ravel()
    x_pred = np.asarray(x_pred).ravel()
    return np.sqrt(np.mean((x_true - x_pred) ** 2))


def rho_jacobi(A):
    """Spectral radius of the Jacobi iteration matrix M = I - D^{-1} A."""
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    d = A.diagonal() if issparse(A) else np.diag(A)
    if np.any(np.abs(d) == 0):
        raise ZeroDivisionError("Matrix has zero on diagonal")
    invd = 1.0 / d
    n = A.shape[0]
    M = identity(n, format='csr') - diags(invd).dot(A) if issparse(A) else np.eye(n) - (A * invd[:, None])
    try:
        val = eigs(M, k=1, which='LM', return_eigenvectors=False)[0]
        return float(abs(val))
    except Exception:
        Mmat = M.toarray() if issparse(M) else np.asarray(M)
        return float(max(np.abs(np.linalg.eigvals(Mmat))))

def compute_omega_opt(A):
    """Asymptotically optimal SOR omega given matrix A."""
    beta = rho_jacobi(A)
    if not (0 <= beta < 1):
        raise ValueError(f"Jacobi spectral radius beta={beta:.6g} not in [0,1).")
    return 1.0 + (beta / (1.0 + np.sqrt(1.0 - beta**2)))**2


# def estimate_asymptotic_cr(residuals, last=10):
#     """Estimate asymptotic linear factor from residual history (normalized norms).
#     Takes geometric mean of last `last` successive ratios r_{k+1}/r_k.
#     """
#     r = np.asarray(residuals, dtype=float)
#     if r.size < 2:
#         return 0.0
#     # take the last `last` ratios (or all if fewer)
#     ratios = r[1:] / r[:-1]
#     ratios = ratios[-last:]
#     # geometric mean (robust to sign is not needed as residuals are norms >=0)
#     # avoid zeros: if any zero ratio, return 0
#     if np.any(ratios == 0):
#         return 0.0
#     return float(np.prod(ratios) ** (1.0 / ratios.size))


def estimate_asymptotic_cr(residuals):
    """Return the convergence rate of the last iteration: -log(r_n / r_{n-1}).

    If the previous residual is zero or the ratio is non-positive, return np.inf.
    """
    r = np.asarray(residuals, dtype=float)
    if r.size < 2:
        return 0.0
    prev = r[-2]
    curr = r[-1]
    if prev == 0:
        return float(np.inf)
    ratio = curr / prev
    if ratio <= 0:
        return float(np.inf)
    return float(-np.log(ratio))




# Define the debug function
def debug_print(debug, *args):
    if debug:
        print(*args)

def flatten_list(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)  # Recursively flatten lists
        else:
            yield item

def check_solution_error_numpy(A, s, b):
    """
    Calculate the residual error ||b - As|| using numpy
    """
    print(f'A shape: {A.shape}\nb shape: {b.shape}\ns shape: {s.shape}') 
    residual = np.linalg.norm(b.reshape(-1, 1) - A.dot(s.reshape(-1, 1))) 
    print(f'residual for 1 numpy: {residual}')

def check_solution_error_torch(A, s, b):
    """
    Calculate the residual error ||b - As|| after linear system is converted to graph
    """
    s = torch.tensor(s.reshape(-1,1)).to(device)
    b = torch.tensor(b.reshape(-1,1)).to(device)
    residual_ = torch.linalg.norm(b - torch.sparse.mm(A, s))
    print('Residual error 2 torch: ',residual_)
    del A, residual_

from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh, norm

def classify_sparse_matrix(A, tol=0):
    """Return dict: {'square', 'symmetric', 'spd', 'psd', 'min_eig'}."""
    if A.shape[0] != A.shape[1]:
        return {'square':False,'symmetric':False,'spd':False,'psd':False,'min_eig':None}
    # relative symmetry check using infinity-norm
    rel = norm(A - A.T, ord=np.inf)
    Ainf = max(1.0, norm(A, ord=np.inf))
    symmetric = (rel <= tol * Ainf)
    if not symmetric:
        return {'square':True,'symmetric':False,'spd':False,'psd':False,'min_eig':None}
    # compute smallest algebraic eigenvalue (sparse) with dense fallback for robustness
    try:
        lam = float(eigsh(A, k=1, which='SA', return_eigenvectors=False)[0])
    except Exception:
        M = A.toarray() if issparse(A) else np.asarray(A)
        lam = float(np.linalg.eigvalsh(M).min())
    spd = lam > tol
    psd = lam >= -tol
    return {'square':True,'symmetric':True,'spd':spd,'psd':psd,'min_eig':lam}


def plot_omega_over_episodes(variable_list, xlabel, ylabel, label, name, optimal_omega=None, window=25, log=False):
    """
    Plot omega (single solid line = rolling mean) with shaded fluctuation band (mean ± std).
    Theoretical omega (optimal_omega) is shown as a dotted red horizontal line.
    """
    import pandas as pd
    fontsize = 20

    x = np.asarray(variable_list, dtype=float).ravel()
    n = x.size

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor('#F5F5F5')

    if n:
        s = pd.Series(x)

        # rolling mean/std (min_periods=1 ensures short series work)
        ma = s.rolling(window=window, min_periods=1).mean().to_numpy()
        std = s.rolling(window=window, min_periods=1).std(ddof=0).to_numpy()
        std = np.nan_to_num(std, nan=0.0)

        # force the first plotted mean to equal the actual first omega and zero fluctuation
        ma[0] = x[0]
        std[0] = 0.0

        lower = ma - std
        # clip lower to positive epsilon to avoid problems with log scale
        lower = np.maximum(lower, 1e-8)
        upper = ma + std

        idx = np.arange(n)

        # plot shaded fluctuation band first, then the main solid line on top
        ax.fill_between(idx, lower, upper, alpha=0.25, linewidth=0)
        ax.plot(idx, ma, linewidth=3, label=label, zorder=5)

    else:
        ax.text(0.5, 0.5, 'no data', ha='center', fontsize=fontsize)

    # theoretical omega as dotted red horizontal line
    if optimal_omega is not None:
        ax.axhline(y=optimal_omega, linestyle='--', linewidth=3,
                   label=f'optimal ω={optimal_omega:.2f}', color='red', zorder=6)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(fontsize= fontsize)
    ax.grid(True, linestyle='-', linewidth=0.6, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    if log==True:
        ax.set_yscale('log')
        name = f'{name}_log.png'

    plt.tight_layout()
    plt.savefig(name, bbox_inches="tight", dpi=300)
    plt.close(fig)



def plot_rewards(episode_rewards, path, is_ipython, show_result=False, window=20, name=None):
    """Plot moving mean (single solid line) with shaded band; larger fonts and grid lines."""
    os.makedirs(path, exist_ok=True)
    r = np.asarray(episode_rewards, dtype=float)

    # font settings
    tick_fs = 20
    label_fs = 20
    title_fs = 20
    
    plt.ioff()
    plt.figure(figsize=(8, 6))
    plt.clf()
    plt.title('Result' if show_result else 'Training - Rewards', fontsize=title_fs)
    plt.xlabel('Episode', fontsize=label_fs)
    plt.ylabel('Reward', fontsize=label_fs)

    if r.size:
        import pandas as pd
        s = pd.Series(r)

        # rolling mean and std (min_periods=1 handles short arrays)
        ma = s.rolling(window=window, min_periods=1).mean().to_numpy()
        std = s.rolling(window=window, min_periods=1).std(ddof=0).to_numpy()
        std = np.nan_to_num(std, nan=0.0)

        # ensure first plotted value equals the actual first reward
        ma[0] = r[0]
        std[0] = 0.0

        lower = ma - std
        upper = ma + std
        x = np.arange(len(r))

        # shaded region then single solid line (mean)
        plt.fill_between(x, lower, upper, alpha=0.25)
        plt.plot(x, ma, linewidth=2)

        # optional: keep ticks sensible if many episodes
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)

        # grid lines
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    else:
        plt.text(0.5, 0.5, 'no data', ha='center', fontsize=label_fs)

    plt.tight_layout()
    if name is not None:
        plt.savefig(os.path.join(path, f"{name}"), dpi=300)
        plt.close()
    else:
        plt.savefig(os.path.join(path, "training_rewards.png"), dpi=300)
        plt.close()





def plot_durations(episode_durations, path, is_ipython, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.savefig(f"{path}/training_durations.png")
    # plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     if not show_result:
    #         display.display(plt.gcf())
    #         display.clear_output(wait=True)
    #     else:
    #         display.display(plt.gcf())
    plt.ioff()
    plt.close()

def plot_results_dynamic(all_residuals_dict, N, rl_omega=None, opt_omega=None, save_path=None, target_tol=None):
    def _cr(resid):
        if not resid or len(resid) < 2: return float('nan')
        r0, rend = float(resid[0]), float(resid[-1])
        if r0 <= 0 or rend <= 0: return float('nan')
        return - (np.log(rend) - np.log(r0)) / float(len(resid))

    # slightly taller figure so omega-value annotations don't get clipped
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1]})
    handles, labels = [], []

    # ---------- residual plots (top) ----------
    baseline = all_residuals_dict.get('residuals_baseline', {}).get(N)
    if baseline:
        h_baseline, = ax.semilogy(range(1, len(baseline) + 1), baseline,
                                  color='k', linewidth=3, label='baseline')
        handles.append(h_baseline); labels.append('baseline')
    else:
        h_baseline, = ax.plot([], [], color='k', linewidth=3, label='baseline')
        handles.append(h_baseline); labels.append('baseline')

    sor_dict = all_residuals_dict.get('residuals_SOR', {}).get(N, {})
    sor_items = [(float(o), sor_dict[o]) for o in sor_dict.keys()]
    sor_items = sorted(sor_items, key=lambda x: x[0])
    if sor_items:
        min_iter_omega = min((o for o, r in sor_items if r), key=lambda o: len(sor_dict[o]))
        crs = {o: _cr(sor_dict[o]) for o, _ in sor_items if sor_dict[o]}
        max_cr_omega = max(crs, key=lambda o: crs[o]) if crs else None
    else:
        min_iter_omega = max_cr_omega = None

    opt_plotted = False; min_plotted = False; max_plotted = False
    background_res = []

    for omega, res in sor_items:
        if not res:
            continue
        is_opt = (opt_omega is not None and np.isclose(omega, float(opt_omega), atol=1e-8))
        is_min = (min_iter_omega is not None and np.isclose(omega, float(min_iter_omega), atol=1e-8))
        is_max = (max_cr_omega is not None and np.isclose(omega, float(max_cr_omega), atol=1e-8))

        if is_opt:
            h_opt, = ax.semilogy(range(1, len(res) + 1), res,
                                 linestyle='-.', color='green', alpha=0.9, linewidth=3, zorder=18)
            if not opt_plotted:
                lbl = f'$M_{{SOR}}$ with $\\omega={opt_omega:.3f}$ (theoretical)'
                handles.append(h_opt); labels.append(lbl)
                opt_plotted = True
        elif is_min:
            h_min, = ax.semilogy(range(1, len(res) + 1), res,
                                 linewidth=3, linestyle=':', color='magenta', alpha=0.85, zorder=16)
            if not min_plotted:
                lbl = f'$M_{{SOR}}$ with $\\omega={min_iter_omega:.3f}$ (min iter)'
                handles.append(h_min); labels.append(lbl)
                min_plotted = True
        elif is_max:
            h_max, = ax.semilogy(range(1, len(res) + 1), res,
                                 linewidth=3, linestyle='--', color='orange', alpha=0.9, zorder=14)
            if not max_plotted:
                lbl = f'$M_{{SOR}}$ with $\\omega={max_cr_omega:.3f}$ (max $\\rho$)'
                handles.append(h_max); labels.append(lbl)
                max_plotted = True
        else:
            background_res.append(res)

    if background_res:
        max_len = max(len(r) for r in background_res)
        arr = np.full((len(background_res), max_len), np.nan)
        for i, r in enumerate(background_res):
            arr[i, :len(r)] = r
        min_vals = np.nanmin(arr, axis=0)
        max_vals = np.nanmax(arr, axis=0)
        valid = ~np.isnan(min_vals) & ~np.isnan(max_vals)
        if valid.any():
            x = np.arange(1, max_len + 1)[valid]
            ax.fill_between(x, min_vals[valid], max_vals[valid],
                            color='lightgray', alpha=0.5, zorder=1)

    rl = all_residuals_dict.get('residuals_RL', {}).get(N)
    if rl:
        rl_label = 'Dynamic $M_{SOR}$ (ours)'
        h_rl, = ax.semilogy(range(1, len(rl) + 1), rl,
                            linestyle='--', color='blue', linewidth=3, zorder=20)
        handles.append(h_rl); labels.append(rl_label)
    else:
        rl_label = None

    # Ensure optimal omega line is visible and always labeled (draw horizontal if not already plotted)
    opt_label_str = f'$M_{{SOR}}$ with $\\omega={opt_omega:.3f}$ (theoretical)' if opt_omega is not None else '$M_{SOR}$ with $\\omega=\\mathrm{nan}$ (theoretical)'
    if opt_omega is not None and not opt_plotted:
        h_optline = ax.axhline(y=float(opt_omega), linestyle='-.', color='green', linewidth=3, alpha=0.9, zorder=19)
        handles.append(h_optline); labels.append(opt_label_str)

    # dotted, noticeable target tolerance line (tab:red) and add to legend (shows numeric value)
    if target_tol is not None:
        h_tol = ax.axhline(y=float(target_tol), color='tab:red', linestyle=':', linewidth=1.8, alpha=0.9, zorder=17)
        lbl_tol = f'target tol = {float(target_tol):.3e}'
        handles.append(h_tol); labels.append(lbl_tol)

        # also ensure the y-axis includes the target value (so it's visible on axis scale)
        ymin, ymax = ax.get_ylim()
        if target_tol < ymin:
            ax.set_ylim(target_tol, ymax)
        elif target_tol > ymax:
            ax.set_ylim(ymin, target_tol * 1.2)

    # Controlled legend order (use the exact labels we appended)
    desired_order = [
        'baseline',
        'Dynamic $M_{SOR}$ (ours)',
        f'$M_{{SOR}}$ with $\\omega={min_iter_omega:.3f}$ (min iter)' if min_iter_omega is not None else None,
        f'$M_{{SOR}}$ with $\\omega={max_cr_omega:.3f}$ (max $\\rho$)' if max_cr_omega is not None else None,
        opt_label_str,
        (lbl_tol if target_tol is not None else None)
    ]
    label2handle = dict(zip(labels, handles))
    final_handles = [label2handle[l] for l in desired_order if l in label2handle]
    final_labels = [l for l in desired_order if l in label2handle]

    # ---------- formatting: make fonts and ticks consistent ----------
    fontsize = 20
    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Residual norm', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
    if final_handles:
        ax.legend(final_handles, final_labels, loc='upper right', fontsize=12)

    # ---------- omega vs iteration (bottom) (slimmer subplot) ----------
    if rl_omega is not None:
        omega_arr = np.asarray(rl_omega, dtype=float).ravel()
        if omega_arr.size:
            x_omega = np.arange(1, omega_arr.size + 1)
            ax2.plot(x_omega, omega_arr, linestyle='-', linewidth=2, color='orange', zorder=5)
            ax2.set_ylabel('omega', fontsize=fontsize)
            ax2.set_xlabel('Iteration', fontsize=fontsize)
            # expand y-limits slightly so annotations are not clipped at the top/bottom
            ax2.set_ylim(-0.05, 2.05)
            ax2.tick_params(axis='x', labelsize=fontsize)
            ax2.tick_params(axis='y', labelsize=fontsize)
            ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

            # annotate omega values (black text, two decimal places) along the line.
            # annotate every step to avoid clutter; choose ~20 labels max
            step = max(1, len(omega_arr) // 20)
            for i in range(0, len(omega_arr), step):
                # place annotations slightly above/below point to avoid overlap with axis
                y = omega_arr[i]
                va = 'bottom' if (y < 1.9) else 'top'
                ax2.text(x_omega[i], y, f'{y:.2f}',
                         color='black', fontsize=10, ha='center', va=va,
                         rotation=45, bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8, edgecolor='none'),
                         zorder=10, clip_on=False)
        else:
            ax2.text(0.5, 0.5, 'no omega data', ha='center', va='center', fontsize=fontsize)
            ax2.tick_params(axis='both', labelsize=fontsize)
    else:
        ax2.text(0.5, 0.5, 'no omega data', ha='center', va='center', fontsize=fontsize)
        ax2.tick_params(axis='both', labelsize=fontsize)

    # save and close
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig, ax







def plot_results(all_residuals_dict, N, rl_omega=None, opt_omega=None, save_path=None):
    def _cr(resid):
        if not resid or len(resid) < 2: return float('nan')
        r0, rend = float(resid[0]), float(resid[-1])
        if r0 <= 0 or rend <= 0: return float('nan')
        return - (np.log(rend) - np.log(r0)) / float(len(resid))

    fig, ax = plt.subplots(figsize=(8,6))
    handles, labels = [], []

    # baseline (unchanged)
    baseline = all_residuals_dict.get('residuals_baseline', {}).get(N)
    if baseline:
        h_baseline, = ax.semilogy(range(1,len(baseline)+1), baseline, color='k', linewidth=3, label='baseline')
        handles.append(h_baseline); labels.append('baseline')

    # SOR dict
    sor_dict = all_residuals_dict.get('residuals_SOR', {}).get(N, {})
    sor_items = [(float(o), sor_dict[o]) for o in sor_dict.keys()]
    sor_items = sorted(sor_items, key=lambda x: x[0])
    if sor_items:
        min_iter_omega = min((o for o,r in sor_items if r), key=lambda o: len(sor_dict[o]))
        crs = {o: _cr(sor_dict[o]) for o,_ in sor_items if sor_dict[o]}
        max_cr_omega = max(crs, key=lambda o: crs[o]) if crs else None
    else:
        min_iter_omega = max_cr_omega = None

    # plot SOR lines: draw only the special lines, collect non-special into background_res
    opt_plotted = False; min_plotted = False; max_plotted = False
    background_res = []

    for omega, res in sor_items:
        if not res:
            continue
        is_opt = (opt_omega is not None and np.isclose(omega, float(opt_omega), atol=1e-8))
        is_min = (min_iter_omega is not None and np.isclose(omega, float(min_iter_omega), atol=1e-8))
        is_max = (max_cr_omega is not None and np.isclose(omega, float(max_cr_omega), atol=1e-8))

        if is_opt:
            h_opt, = ax.semilogy(range(1,len(res)+1), res, linestyle='-.', color='green', alpha=0.8, linewidth=2.2, zorder=18)
            if not opt_plotted:
                handles.append(h_opt); labels.append(f'optimal omega={opt_omega:.3f}')
                opt_plotted = True
        elif is_min:
            h_min, = ax.semilogy(range(1,len(res)+1), res, linewidth=2, linestyle=':', color='magenta', alpha=0.7, zorder=14)
            if not min_plotted:
                handles.append(h_min); labels.append(f'min iterations (omega={min_iter_omega:.3f})')
                min_plotted = True
        elif is_max:
            h_max, = ax.semilogy(range(1,len(res)+1), res, linewidth=2, linestyle='--', color='orange', alpha=0.8, zorder=16)
            if not max_plotted:
                handles.append(h_max); labels.append(f'max cr (omega={max_cr_omega:.3f})')
                max_plotted = True
        else:
            # collect background residuals (do not plot each line)
            background_res.append(res)

    # merge background lines into an envelope and shade the area (behind main lines)
    if background_res:
        max_len = max(len(r) for r in background_res)
        arr = np.full((len(background_res), max_len), np.nan)
        for i, r in enumerate(background_res):
            arr[i, :len(r)] = r
        min_vals = np.nanmin(arr, axis=0)
        max_vals = np.nanmax(arr, axis=0)
        valid = ~np.isnan(min_vals) & ~np.isnan(max_vals)
        if valid.any():
            x = np.arange(1, max_len+1)[valid]
            ax.fill_between(x, min_vals[valid], max_vals[valid],
                            color='lightgray', alpha=0.5, zorder=1)

    # RL: dotted blue, in front (unchanged)
    rl = all_residuals_dict.get('residuals_RL', {}).get(N)
    if rl:
        rl_label = f'learned SOR (omega={float(rl_omega[0]):.3f})' if rl_omega is not None else 'learned SOR'
        h_rl, = ax.semilogy(range(1,len(rl)+1), rl, linestyle='--', color='blue', linewidth=2.5, zorder=20)
        handles.append(h_rl); labels.append(rl_label)

    # build controlled legend in requested order
    desired_order = ['baseline',
                     rl_label if rl else None,
                     (f'min iterations (omega={min_iter_omega:.3f})' if min_iter_omega is not None else None),
                     (f'max cr (omega={max_cr_omega:.3f})' if max_cr_omega is not None else None),
                     (f'optimal omega={opt_omega:.3f}' if opt_omega is not None else None)]
    label2handle = dict(zip(labels, handles))
    final_handles = [label2handle[l] for l in desired_order if l in label2handle]
    final_labels = [l for l in desired_order if l in label2handle]

    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Residual norm', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)


    # ax.set_xlabel('Iteration'); ax.set_ylabel('Residual norm')
    # ax.set_title(f'Residual histories (N={N})')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    if final_handles:
        ax.legend(final_handles, final_labels, loc='upper right', fontsize=15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

    return fig, ax



# def plot_results(all_residuals_dict, save_path, episode=None, N=None, default_omega=None, test_omega=None, mode='train', omega_list=None):

#     # create main plot + a small axis below for omega vs iteration
#     # fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
#     # ax.set_facecolor('#F5F5F5')
#     fontsize = 20

#     if mode == 'train':

#         fig, (ax, ax2) = plt.subplots(
#             2, 1, figsize=(8, 7),
#             gridspec_kw={'height_ratios': [4, 1]},
#             sharex=True
#         )
#         ax2.set_facecolor('#FFFFFF')   # or any background you prefer for omega subplot
#         file_name = f'{save_path}/residual_plot_{episode}_N{N}.png'
        
#         data_labels = {
#             'residuals_RL': f'ours(ep={episode})',
#             'residuals_SOR': f'w_opt={default_omega:.2f}',
#             'residuals_baseline': 'baseline'
#         }

#     elif mode == 'test':

#         # testing mode: single plot (no omega subplot)
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#         ax2 = None  # sentinel so you can safely check later

#         file_name = f'{save_path}/residual_plot_N{N}.png'
        
#         data_labels = {
#             'residuals_RL': f'ours (test_omega={test_omega:.2f})',
#             'residuals_SOR': f'w_opt={default_omega:.2f}',
#             'residuals_baseline': 'baseline'
#         }


#     # choose easily distinguishable colors and increased line widths
#     colors = {'residuals_RL': 'blue', 'residuals_SOR': 'red', 'residuals_baseline': 'green'}
#     widths = {'residuals_RL': 3.0, 'residuals_SOR': 3.0, 'residuals_baseline': 3.0}
#     styles = {'residuals_RL': '-', 'residuals_SOR': '--', 'residuals_baseline': '-.'}
#     alphas = {'residuals_RL': 1.0, 'residuals_SOR': 0.5, 'residuals_baseline': 0.6}

#     for key, label in data_labels.items():
#         data = all_residuals_dict.get(key, {}).get(N)
#         if data:
#             ax.plot(data, label=label, color=colors.get(key, None), linewidth=widths.get(key, 2.0),
#                     linestyle=styles.get(key, '-'), alpha=alphas.get(key, 0.9))

#     ax.set_xlabel('Iterations', fontsize=fontsize)
#     ax.set_ylabel('Residual Norm', fontsize=fontsize)
#     ax.legend(fontsize=fontsize)
#     ax.grid(color='#D3D3D3', linestyle='-', linewidth=0.5)
#     ax.tick_params(axis='both', labelsize=fontsize)
#     ax.set_yscale('log')
#     ax.set_title(f'Size: N = {N}', fontsize=fontsize)

#     # plot omega_list below the main plot (aligned with iterations), show values with 2 decimals
#     if omega_list is not None:
#         rl_data = all_residuals_dict.get('residuals_RL', {}).get(N)
#         n_rl = len(rl_data) if rl_data is not None else None
#         npoints = min(len(omega_list), n_rl) if n_rl is not None else len(omega_list)
#         if npoints > 0:
#             x = list(range(npoints))
#             y = [omega_list[i] for i in x]
#             ax2.set_facecolor('#F5F5F5')  # light gray background
#             ax2.plot(x, y, marker='o', linestyle='-', linewidth=1.2, markersize=5, color='tab:orange', alpha=0.9)
#             # annotate each point with 2 decimals (numbers in black for visibility)
#             for xi, yi in zip(x, y):
#                 ax2.text(xi, yi, f'{yi:.2f}', ha='center', va='bottom', fontsize=8, rotation=45, color='black')
#             ax2.set_ylabel('omega', fontsize=fontsize-6)
#             ax2.tick_params(axis='both', labelsize=fontsize-6)
#             ax2.set_ylim(0, 2)  # force y-axis between 0 and 2
#             ax2.grid(color='#D3D3D3', linestyle='-', linewidth=0.5)


#     fig.tight_layout()
#     fig.savefig(file_name, bbox_inches="tight", dpi=300)
#     plt.close(fig)





def plot_results_no_RL(all_residuals_dict, save_path, data_size_list, default_omega):
    """
    Plot and save the residual curves for different methods.

    Parameters:
        all_residuals_dict (dict): Dictionary containing residuals data.
        save_path (str): Directory where the plot will be saved.
        episode (int): Current episode number (used in labels).
        N (int): Key/index used to access residual data.
        default_omega (float): Parameter for the default SOR method.
        time_list (optional): Not used in this implementation.
        plot_cpu_time (bool, optional): Not used in this implementation.
    """

    for N in data_size_list:
        file_name = f'{save_path}/residual_plot_N{N}.png'
        fontsize = 20

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_facecolor('#F5F5F5')


        # Mapping from dictionary keys to labels
        data_labels = {
            'residuals_SOR': f'SOR w_opt={default_omega:.2f}',
            'residuals_baseline': f'Baseline'
        }

        # Plot each dataset if available
        for key, label in data_labels.items():
            data = all_residuals_dict.get(key, {}).get(N)
            if data:
                ax.plot(data, label=label, linewidth=3.0)

        # Set labels, grid, and tick parameters

        ax.set_xlabel('Iterations', fontsize=fontsize)
        ax.set_ylabel('Residual Norm', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.grid(color='#D3D3D3', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_yscale('log')

        # Add title with size N
        ax.set_title(f'Size: N = {N}', fontsize=fontsize)

        # Save and close the figure
        fig.savefig(file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)


def load_policy_net(policy_net, filepath):
    policy_net.load_state_dict(torch.load(filepath))
    policy_net.eval()  # Set the network to evaluation mode
    print("Policy network weights loaded successfully.")
    return policy_net

def parse_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Ignore empty lines or lines that start with a comment
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split the line into key and value
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Convert the value to the appropriate type
            if value.isdigit():
                value = int(value)  # Convert to integer
            elif value.replace('.', '', 1).isdigit() and '.' in value:
                value = float(value)  # Convert to float
            elif value.lower() == 'true':
                value = True  # Convert to boolean True
            elif value.lower() == 'false':
                value = False  # Convert to boolean False
            elif value.startswith('[') and value.endswith(']'):
                # Convert to a list (assumes list of integers)
                value = eval(value)  # Use eval cautiously; safer alternatives exist
            elif value.lower() == 'none':
                value = None  # Convert to None
            else:
                value = str(value)  # Keep as string

            config[key] = value

    return config


import os, numpy as np, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, eigs, ArpackNoConvergence

def analyze_matrix(A, out_prefix, nev=50):
    A = csr_matrix(A)
    N = A.shape[0]
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

    # 1) estimate condition number
    try:
        λmax = eigsh(A, 1, which='LM', return_eigenvectors=False)[0]
        λmin = eigsh(A, 1, which='SM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        vals = np.abs(eigs(A, 2, which='BE', return_eigenvectors=False))
        λmax, λmin = vals.max(), vals.min()
    κ = abs(λmax/λmin)
    print(f"cond(A) ≈ {κ:.2e}")

    # 2) compute a sample of eigenvalues
    try:
        vals = np.real(eigs(A, min(nev, A.shape[0]-2), return_eigenvectors=False))
    except ArpackNoConvergence as e:
        vals = np.real(e.eigenvalues)
    vals = np.sort(vals)[::-1]  # sort descending: largest first

    # 3) one figure with 3 subplots: scatter, histogram, spy
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    # scatter of sorted eigenvalues
    axs[0].plot(vals, '.', ms=4)
    axs[0].set(title=f'Eigenvalues κ(A)≈{κ:.2e}', xlabel='i', ylabel='λᵢ')

    # histogram with edgecolors
    axs[1].hist(vals, bins=20, edgecolor='black')
    axs[1].set(title='Histogram of λᵢ', xlabel='λ', ylabel='Count')

    # spy plot
    axs[2].spy(A, markersize=0.5)
    axs[2].set(title='Sparsity (spy)')

    plt.tight_layout()
    plt.savefig(f"{out_prefix}/N{N}_analysis.png", dpi=400)
    plt.close(fig)

    return κ