import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def _policy_forward(policy_net, states, device):
    """Run policy_net on batch states and return action scores/probs (numpy)."""
    policy_net.eval()
    with torch.no_grad():
        s_t = torch.tensor(states, dtype=torch.float32, device=device)
        # ensure batch dimension
        if s_t.ndim == 1:
            s_t = s_t.unsqueeze(0)
        out = policy_net(s_t)
        # policy_net might return (logits,) or (q_values,) or probabilities
        if isinstance(out, tuple) or isinstance(out, list):
            out = out[0]
        out = out.cpu().numpy()
    return out  # shape (batch, n_actions) or (batch,)

def plot_policy_heatmap(policy_net, omega_values,
                        rho_bounds=(0.0, 1.0), iter_bounds=(0, 2464),
                        grid_size=(200, 200), device='cpu',
                        state_transform=None, savepath=None, cmap='viridis'):
    """
    Decision map over (rho, n_iter). Iterations run from iter_bounds[0] (typically 0) to iter_bounds[1] (N).
    Colorbar shows ω on a continuous 0..2 scale (not every discrete ω label).
    omega_values should come from np.linspace(0.0, 2.0, num=n_actions+2)[1:-1].
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # grid in rho (x) and n_iter (y)
    rho_lin = np.linspace(rho_bounds[0], rho_bounds[1], grid_size[1])
    iter_lin = np.linspace(iter_bounds[0], iter_bounds[1], grid_size[0])
    R, I = np.meshgrid(rho_lin, iter_lin)               # shape (ny, nx)
    pts = np.column_stack([R.ravel(), I.ravel()])

    if state_transform is not None:
        pts = state_transform(pts)

    # forward pass
    policy_net.eval()
    with torch.no_grad():
        s = torch.tensor(pts, dtype=torch.float32, device=device)
        out = policy_net(s)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.cpu().numpy()

    # get action indices and map to omega values
    if out.ndim == 1:
        act_idx = out.astype(int)
    else:
        act_idx = np.argmax(out, axis=1)
    act_idx = act_idx.reshape(R.shape)                 # (ny, nx)
    omega_map = np.asarray(omega_values)[act_idx]      # map each grid cell to its omega

    # plot omega_map with a continuous colormap from 0 (excluded) to 2 (excluded)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(omega_map, origin='lower',
                   extent=(rho_bounds[0], rho_bounds[1], iter_bounds[0], iter_bounds[1]),
                   aspect='auto', cmap=cmap, vmin=0.0, vmax=2.0, interpolation='nearest')

    ax.set_xlabel('Convergence rate $\\rho$')
    ax.set_ylabel('Number of iterations $n$')
    ax.set_title('Policy decision map (ω)')

    # continuous colorbar showing range 0..2 (choose a few ticks, not every omega)
    ticks = np.linspace(0.0, 2.0, 5)  # e.g. [0, 0.5, 1.0, 1.5, 2.0]
    cbar = fig.colorbar(im, ax=ax, ticks=ticks, pad=0.02)
    cbar.set_label('ω', rotation=270, labelpad=15)

    ax.grid(False)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig, ax



def plot_policy_with_trajectories(policy_net,
                                  observed_states,   # Nx2 array-like of (rho, n_iter)
                                  observed_actions,  # length-N array of action indices
                                  omega_values,
                                  device='cpu',
                                  state_transform=None,
                                  grid_kwargs=None,
                                  traj_kwargs=None,
                                  savepath=None):
    """
    Plot decision map and overlay observed (state,action) points.
    """
    grid_kwargs = grid_kwargs or {}
    traj_kwargs = traj_kwargs or {}

    # set bounds from observed data with small margins
    s = np.asarray(observed_states)
    rho_min, rho_max = s[:,0].min(), s[:,0].max()
    n_min, n_max = s[:,1].min(), s[:,1].max()
    # add margin
    rho_pad = max(1e-6, 0.05*(rho_max - rho_min) if rho_max>rho_min else 0.1)
    n_pad = max(1.0, 0.05*(n_max - n_min) if n_max>n_min else 5)
    rho_bounds = (max(0.0, rho_min - rho_pad), rho_max + rho_pad)
    iter_bounds = (max(1, n_min - n_pad), n_max + n_pad)

    gk = dict(rho_bounds=rho_bounds, iter_bounds=iter_bounds)
    gk.update(grid_kwargs or {})

    fig, ax = plot_policy_heatmap(policy_net, omega_values, device=device,
                                 state_transform=state_transform, savepath=None, **gk)

    # overlay observations: scatter colored by observed_actions
    obs = np.asarray(observed_states)
    act = np.asarray(observed_actions)
    cmap_list = plt.get_cmap('tab10', len(omega_values))
    sc = ax.scatter(obs[:,0], obs[:,1], c=act, cmap=cmap_list, edgecolor='k', s=30, alpha=0.9, **traj_kwargs)
    # create legend for scatter (small samples)
    handles = []
    for i, omega in enumerate(omega_values):
        if i in np.unique(act):
            handles.append(plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=cmap_list(i),
                                      markeredgecolor='k', label=f'{i}: ω={omega:.3f}', markersize=6))
    if handles:
        ax.legend(handles=handles, bbox_to_anchor=(1.02,1), loc='upper left', title='Observed actions')

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig, ax
