"""Regenerates Figs. 1-5 of the paper from the JSON files in ../results. Run from this folder: python make_figures.py"""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

P = {i: json.load(open(f'../results/results_{i + 1:02d}.json')) for i in range(5)}
BLUE, ORANGE, AQUA, YEL = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'legend.frameon': False, 'savefig.dpi': 400})
W = 3.45

def clean(ax, grid_axis='y'):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5); ax.set_axisbelow(True)

# ---------- Fig 1: schematic of the two landmark designs ----------
fig, axes = plt.subplots(2, 1, figsize=(W, 2.9), sharex=True)
stays = [('Septic stay', 0, 62, True), ('Non-septic, short', 0, 30, False), ('Non-septic, long', 0, 70, False),
         ('Non-septic, short', 0, 22, False)]
t_case = 50; h = 6
for k, ax in enumerate(axes):
    for i, (lab, a, b, sep) in enumerate(stays):
        y = len(stays) - i
        ax.plot([a, b], [y, y], color=ORANGE if sep else BLUE, lw=3, solid_capstyle='round')
        ax.text(-2, y, lab, ha='right', va='center', fontsize=6.5, color=INK)
    ax.axvline(t_case, color=INK2, lw=0.6, ls=':')
    ax.plot([t_case], [4], marker='v', color=INK, ms=5)
    ax.annotate('onset', xy=(t_case + h, 4), xytext=(t_case + h, 4.55), ha='center', fontsize=6, color=INK2,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
    if k == 0:
        for y, t in ((3, 18), (2, 50), (1, 15)):
            ax.plot([t], [y], marker='v', color=INK, ms=5)
        ax.set_title('(a) Time-matched design',
                     fontsize=6.5, loc='left', color=INK)
    else:
        ax.plot([t_case], [2], marker='v', color=INK, ms=5)
        for y in (3, 1):
            ax.text(t_case + 1, y + 0.18, 'not at risk at t + h', fontsize=5.8, color=INK2)
        ax.axvspan(t_case, t_case + h, color=GRID, alpha=0.6, lw=0)
        ax.set_title('(b) Risk-set design',
                     fontsize=6.5, loc='left', color=INK)
    ax.set_yticks([]); ax.set_ylim(0.4, 4.9)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
axes[1].set_xlabel('Hours since ICU admission'); axes[1].set_xlim(-1, 75)
fig.subplots_adjust(left=0.27, right=0.98, top=0.93, bottom=0.13, hspace=0.45)
fig.savefig('fig1_design.png'); plt.close(fig)

# ---------- Fig 2: timing mechanism ----------
td = P[2]['timing_diag']; q = [10, 25, 50, 75, 90, 99]
fig, ax = plt.subplots(figsize=(W, 2.2))
for s, mk in (('A', 'o'), ('B', 's')):
    ax.plot(q, [td[s]['septic_landmark_quantiles'][str(x)] for x in q], color=ORANGE, marker=mk, ms=4, lw=1.5,
            label=f'Septic landmark time, Set {s}', ls='-' if s == 'A' else '--')
    ax.plot(q, [td[s]['nonseptic_length_quantiles'][str(x)] for x in q], color=BLUE, marker=mk, ms=4, lw=1.5,
            label=f'Non-septic stay length, Set {s}', ls='-' if s == 'A' else '--')
ax.axhline(48, color=INK2, lw=0.6, ls=':'); ax.text(11, 52, 'common-support limit (48 h)', fontsize=6, color=INK2)
ax.set_yscale('log'); ax.set_yticks([10, 20, 50, 100, 200]); ax.set_yticklabels(['10', '20', '50', '100', '200'])
ax.set_xticks(q); ax.set_xlabel('Percentile'); ax.set_ylabel('Hours since ICU admission')
clean(ax); ax.legend(fontsize=6, loc='upper left', ncol=1)
fig.tight_layout(); fig.savefig('fig2_mechanism.png'); plt.close(fig)

# ---------- Fig 3: inflation ladder ----------
p3, p4, p2 = P[3], P[4], P[2]
rows = [('Hourly, random row split (A)', p3['hourly_challenge_style']['random_row_split_A_with_ICULOS']['AUROC'], None, YEL),
        ('Hourly, patient split (A)', p3['hourly_challenge_style']['patient_split_A_with_ICULOS']['AUROC'], None, YEL),
        ('Hourly, hospital split (A→B)', p3['hourly_challenge_style']['hospital_split_AtoB_with_ICULOS']['AUROC'], None, YEL),
        ('Landmark, time-matched (A→B)', p2['robustness']['trunc']['AtoB']['AUROC'], p2['robustness']['trunc']['AtoB']['AUROC_CI'], BLUE),
        ('Landmark, time-matched, ≤48 h (A→B)', p3['common_support_48h']['trunc']['AtoB']['AUROC'], p3['common_support_48h']['trunc']['AtoB']['AUROC_CI'], BLUE),
        ('Risk-set, full window (A→B)', p2['robustness']['riskset_k10']['AtoB']['AUROC'], p2['robustness']['riskset_k10']['AtoB']['AUROC_CI'], ORANGE),
        ('Risk-set, ≤48 h, primary (A→B)', p4['discrimination_6h']['AtoB']['Gradient boosting']['AUROC'], p4['discrimination_6h']['AtoB']['Gradient boosting']['CI'], ORANGE)]
fig, ax = plt.subplots(figsize=(W, 2.4))
y = np.arange(len(rows))[::-1]
for yi, (lab, v, ci, c) in zip(y, rows):
    ax.barh(yi, v - 0.5, left=0.5, height=0.6, color=c)
    if ci: ax.plot(ci, [yi, yi], color=INK, lw=0.8)
    ax.text(max(v, ci[1] if ci else v) + 0.008, yi, f'{v:.3f}', va='center', fontsize=6.5, color=INK)
sirs = p4['discrimination_6h']['AtoB']['SIRS']['AUROC']
ax.axvline(sirs, color=INK2, lw=0.8, ls='--'); ax.text(sirs + 0.006, -0.95, f'SIRS {sirs:.3f}', fontsize=6, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
ax.set_xlim(0.5, 1.0); ax.set_ylim(-1.2, len(rows) - 0.4); ax.set_xlabel('AUROC (gradient boosting)')
clean(ax, 'x'); fig.tight_layout(); fig.savefig('fig3_ladder.png'); plt.close(fig)

# ---------- Fig 4: transportability gap ----------
def delta(h, s):
    d = p4['discrimination_6h'][s]['Gradient boosting'] if h == 6 else p4['horizons_primary'][str(h)][s]['Gradient boosting']
    return d['delta_vs_SIRS'], d['delta_CI']
H = [4, 6, 8, 12]
fig, ax = plt.subplots(figsize=(W, 2.2))
style = {'AtoA': (BLUE, 'o', '-', 'Within hospital A'), 'BtoB': (BLUE, 's', '--', 'Within hospital B'),
         'AtoB': (ORANGE, 'o', '-', 'Transfer A→B'), 'BtoA': (ORANGE, 's', '--', 'Transfer B→A')}
off = {'AtoA': -0.45, 'BtoB': -0.15, 'AtoB': 0.15, 'BtoA': 0.45}
for s, (c, mk, ls, lab) in style.items():
    v = [delta(h, s) for h in H]; x = np.array(H) + off[s]
    ax.errorbar(x, [a for a, _ in v], yerr=[[a - ci[0] for a, ci in v], [ci[1] - a for a, ci in v]], color=c, marker=mk,
                ms=4, ls=ls, lw=1.2, capsize=1.5, elinewidth=0.7, label=lab)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xticks(H); ax.set_xlabel('Prediction horizon (h)'); ax.set_ylabel('ΔAUROC, gradient boosting − SIRS')
clean(ax); ax.legend(fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); fig.savefig('fig4_gap.png'); plt.close(fig)

# ---------- Fig 5: target label budget ----------
cf = p4['conformal_primary']; B = [10, 20, 30, 50, 100]
fig, axes = plt.subplots(1, 2, figsize=(W, 1.9))
for d, c, mk, lab in (('AtoB', BLUE, 'o', 'A→B'), ('BtoA', ORANGE, 's', 'B→A')):
    mean = [cf[d][f'target_budget_{b}_a0_0.1']['septic_cov'][0] for b in B]
    p10 = [cf[d][f'target_budget_{b}_a0_0.1']['septic_cov_p10'][0] for b in B]
    frac = [cf[d][f'target_budget_{b}_a0_0.1']['frac_draws_cov_ge_085'][0] for b in B]
    axes[0].plot(B, mean, color=c, marker=mk, ms=3.5, lw=1.2, label=f'{lab}, mean')
    axes[0].plot(B, p10, color=c, marker=mk, ms=3.5, lw=1.2, ls='--', label=f'{lab}, 10th pct')
    axes[0].axhline(cf[d]['source_mondrian_a0_0.1']['septic_cov'][0], color=c, lw=0.7, ls=':')
    axes[1].plot(B, frac, color=c, marker=mk, ms=3.5, lw=1.2, label=lab)
axes[0].axhline(0.9, color=INK2, lw=0.6)
axes[0].set_ylabel('Septic coverage'); axes[1].set_ylabel('Draws with coverage ≥ 0.85')
for ax in axes:
    ax.set_xscale('log'); ax.set_xticks(B); ax.set_xticklabels([str(b) for b in B]); ax.set_xlabel('Target septic labels (n)')
    clean(ax)
axes[0].set_ylim(0.6, 0.95); axes[1].set_ylim(0.7, 1.0)
axes[0].legend(fontsize=5.2, loc='lower right'); axes[1].legend(fontsize=6, loc='lower right')
fig.tight_layout(w_pad=1.0); fig.savefig('fig5_budget.png'); plt.close(fig)
print('Figures written to the current folder')
