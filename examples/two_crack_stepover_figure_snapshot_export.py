# === Figure-style single snapshot export (does not replace frame exports) ===
from pathlib import Path
from matplotlib.lines import Line2D

# Run this after the presentation-export setup cell above.  It saves
# publication-style four-panel snapshots to a separate directory.
EXPORT_FIGURE_SNAPSHOT = True
EXPORT_FIGURE_SNAPSHOT_TID = None  # None -> use the same _frame_ids as frame export
EXPORT_FIGURE_SNAPSHOT_DPI = 240
EXPORT_FIGURE_SNAPSHOT_FORMATS = ('png', 'pdf')
EXPORT_FIGURE_SNAPSHOT_DIR = Path('images/two_crack_stepover_figure_snapshots')
EXPORT_FIGURE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _add_panel_label(ax, label, *, color='black'):
    ax.text(
        -0.11, 1.035, label,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=14, fontweight='bold', fontfamily='serif',
        color=color, clip_on=False, zorder=60,
    )


def _label_crack_curve(ax, curve, label, t_frac, offset_points):
    curve = np.asarray(curve, dtype=float)
    idx = int(np.clip(round(float(t_frac) * (len(curve) - 1)), 0, len(curve) - 1))
    xy = curve[idx]
    ax.annotate(
        label,
        xy=(xy[0], xy[1]), xycoords='data',
        xytext=offset_points, textcoords='offset points',
        ha='center', va='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                  edgecolor='black', linewidth=0.8, alpha=0.94),
        arrowprops=dict(arrowstyle='-', color='black', lw=1.0,
                        shrinkA=3, shrinkB=3),
        zorder=70,
    )


def _annotate_cracks_in_geometry_panel(ax):
    return None


def _style_crack_geometry_legend(ax):
    colors = globals().get('_CRACK_LINE_COLORS_EXPORT',
                           ('#6b8f71', '#6b8f71'))
    ax.legend(
        handles=[
            Line2D([0], [0], color=colors[0], lw=1.8, ls='--',
                   label=r'$\Gamma_{c0}$'),
            Line2D([0], [0], color=colors[0], lw=2.0,
                   label=r'$\tilde{\Gamma}_{c0}$'),
            Line2D([0], [0], color=colors[1], lw=1.8, ls='--',
                   label=r'$\Gamma_{c1}$'),
            Line2D([0], [0], color=colors[1], lw=2.0,
                   label=r'$\tilde{\Gamma}_{c1}$'),
        ],
        loc='upper right', fontsize=9, ncol=2, handlelength=2.6,
        columnspacing=1.1, framealpha=0.92, edgecolor='0.65',
        fancybox=False,
    )


def _choose_figure_snapshot_tid():
    if EXPORT_FIGURE_SNAPSHOT_TID is not None:
        return int(EXPORT_FIGURE_SNAPSHOT_TID)
    if '_activity_t' in globals() and np.asarray(_activity_t).size:
        return int(np.nanargmax(_activity_t))
    return int(_frame_ids[len(_frame_ids) // 2]) if len(_frame_ids) else 0


def _figure_snapshot_tids():
    if EXPORT_FIGURE_SNAPSHOT_TID is not None:
        return np.array([int(EXPORT_FIGURE_SNAPSHOT_TID)], dtype=int)
    if '_frame_ids' in globals() and len(_frame_ids):
        return np.asarray(_frame_ids, dtype=int)
    return np.array([_choose_figure_snapshot_tid()], dtype=int)


def _cell_edges_from_centers(centers, *, lower=None, upper=None):
    centers = np.asarray(centers, dtype=float).ravel()
    if centers.size == 0:
        lo = 0.0 if lower is None else float(lower)
        hi = 1.0 if upper is None else float(upper)
        return np.array([lo, hi], dtype=float)
    if centers.size == 1:
        if lower is not None and upper is not None:
            return np.array([float(lower), float(upper)], dtype=float)
        width = 1.0
        return np.array([centers[0] - 0.5 * width, centers[0] + 0.5 * width])

    mid = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mid[0] - centers[0])
    last = centers[-1] + (centers[-1] - mid[-1])
    edges = np.concatenate([[first], mid, [last]])
    if lower is not None:
        edges[0] = max(edges[0], float(lower))
    if upper is not None:
        edges[-1] = min(edges[-1], float(upper))
    return edges


def _arc_time_panel_piecewise_constant(ax, mask, arc_length, tid, patch_extent=None):
    if not np.any(mask):
        ax.text(0.5, 0.5, 'no contact nodes', ha='center', va='center',
                transform=ax.transAxes)
        return None
    s_arc = arc_t_per_node[mask] * arc_length
    order = np.argsort(s_arc)
    s_arc = s_arc[order]
    vals = _v_t_mps[mask][order, :]
    s_edges = _cell_edges_from_centers(s_arc, lower=0.0, upper=arc_length)
    t_edges = _cell_edges_from_centers(_t_phys_sw)
    # Rasterize dense filled artists in vector exports to avoid PDF hairline seams.
    pcm = ax.pcolormesh(
        s_edges, t_edges, vals.T, shading='flat',
        cmap=EXPORT_SLIP_CMAP, vmin=0.0, vmax=_slip_vmax,
        rasterized=True, edgecolors='none', linewidth=0.0,
        antialiased=False,
    )
    ax.axhline(_t_phys_sw[tid], color='#00d5ff', lw=1.8)
    if patch_extent is not None:
        for x_marker in patch_extent:
            ax.axvline(x_marker, color='#00d5ff', ls=':', lw=1.0)
    ax.set_ylabel('time (s)')
    ax.set_xlim(0.0, arc_length)
    ax.tick_params(labelsize=8)
    return pcm


def render_two_crack_velocity_figure_snapshot(tid=None):
    """Publication-style four-panel snapshot using the existing export arrays."""
    tid = _choose_figure_snapshot_tid() if tid is None else int(tid)
    tri, vmag = _bulk_mag_at(tid)

    fig = plt.figure(figsize=(13.2, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.0), height_ratios=(1.0, 0.95))
    ax_mesh = fig.add_subplot(gs[0, 0])
    ax_field = fig.add_subplot(gs[0, 1])
    ax_A = fig.add_subplot(gs[1, 0])
    ax_B = fig.add_subplot(gs[1, 1], sharey=ax_A)

    _draw_mesh_geometry_export(ax_mesh)
    _annotate_cracks_in_geometry_panel(ax_mesh)
    _style_crack_geometry_legend(ax_mesh)
    _add_panel_label(ax_mesh, '(a)')

    levels = np.linspace(0.0, _bulk_vmax, 72)
    cf = ax_field.tricontourf(
        tri, vmag, levels=levels,
        cmap=EXPORT_BULK_CMAP, extend='max',
        zorder=-10,
        antialiased=False,
    )
    cf.set_edgecolor('face')
    cf.set_linewidth(0.0)
    ax_field.set_rasterization_zorder(0)
    _draw_true_crack_export(ax_field)

    active = _v_t_mps[:, tid] >= _active_cut
    sticking = _v_t_mps[:, tid] <= _sticking_cut
    if np.any(sticking):
        ax_field.scatter(_contact_xy_true[sticking, 0], _contact_xy_true[sticking, 1],
                         s=28, facecolors='none', edgecolors='#ffd166',
                         linewidths=1.2, zorder=16)
    if np.any(active):
        ax_field.scatter(_contact_xy_true[active, 0], _contact_xy_true[active, 1],
                         s=38, facecolors='#00d5ff', edgecolors='black',
                         linewidths=0.7, zorder=17)

    ax_field.set_aspect('equal')
    ax_field.set_xlim(XMIN, XMAX)
    ax_field.set_ylim(YMIN, YMAX)
    ax_field.set_xlabel('x (km)')
    ax_field.set_ylabel('y (km)')
    _add_panel_label(ax_field, '(b)')
    cbar = fig.colorbar(cf, ax=ax_field, shrink=0.86, pad=0.015)
    cbar.set_label(r'$\Vert v \Vert$ (m/s)')

    pcm_A = _arc_time_panel_piecewise_constant(
        ax_A, mask_A, ARC_A_LENGTH, tid, patch_extent=_patch_extent_A_export)
    _arc_time_panel_piecewise_constant(ax_B, mask_B, ARC_B_LENGTH, tid)
    ax_A.set_title('')
    ax_B.set_title('')
    ax_A.set_xlabel('arc length s (km)')
    ax_B.set_xlabel('arc length s (km)')
    _add_panel_label(ax_A, '(c)')
    _add_panel_label(ax_B, '(d)')
    fig.colorbar(pcm_A, ax=[ax_A, ax_B], shrink=0.90, pad=0.02,
                 label=r'$|v_t|$ (m/s)')

    return fig, tid


if EXPORT_FIGURE_SNAPSHOT:
    _fig_snapshot_paths = []
    _fig_snapshot_tids = _figure_snapshot_tids()
    for _j, _tid in enumerate(_fig_snapshot_tids):
        _fig_snapshot, _fig_snapshot_tid = render_two_crack_velocity_figure_snapshot(int(_tid))
        _time_tag = f'{_t_phys_sw[_fig_snapshot_tid]:.3f}'.replace('.', 'p')
        _fig_snapshot_stem = (
            EXPORT_FIGURE_SNAPSHOT_DIR
            / f'figure_snapshot_{_j:04d}_t{_time_tag}s'
        )
        for _fmt in EXPORT_FIGURE_SNAPSHOT_FORMATS:
            _fmt = str(_fmt).lower().lstrip('.')
            _fig_snapshot_path = _fig_snapshot_stem.with_suffix(f'.{_fmt}')
            _fig_snapshot.savefig(_fig_snapshot_path, dpi=EXPORT_FIGURE_SNAPSHOT_DPI,
                                  facecolor='white', bbox_inches='tight')
            _fig_snapshot_paths.append(_fig_snapshot_path)
        plt.close(_fig_snapshot)
    print(f'Wrote {len(_fig_snapshot_paths)} figure-style snapshots '
          f'to {EXPORT_FIGURE_SNAPSHOT_DIR}')
