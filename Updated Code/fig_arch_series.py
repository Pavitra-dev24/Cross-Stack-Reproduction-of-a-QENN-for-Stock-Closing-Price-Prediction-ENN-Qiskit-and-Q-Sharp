import sys; sys.path.insert(0,'/home/claude/work')
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import data as D
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})

# ---- Fig 1: data flow of Elman network (a) and QENN (b)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.1))
def box(ax, x, y, w, h, txt, fc='#f2f4fb'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec='#333', lw=1))
    ax.text(x + w/2, y + h/2, txt, ha='center', va='center', fontsize=7.6)
def arr(ax, p, q, txt=None, rad=0.0, dy=0.12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=1, color='#222', connectionstyle=f"arc3,rad={rad}"))
    if txt: ax.text((p[0]+q[0])/2, (p[1]+q[1])/2 + dy, txt, ha='center', fontsize=7)
for ax in axs: ax.set_xlim(0, 10); ax.set_ylim(0, 2.1); ax.axis('off')
ax = axs[0]; ax.set_title('(a) Elman network (classical baselines)', fontsize=8.5, loc='left')
box(ax, 0.2, 1.0, 1.9, 0.8, 'window $x(k)\\in\\mathbb{R}^6$'); box(ax, 3.4, 1.0, 1.9, 0.8, 'hidden $u(k)$\n$\\tanh$ units'); box(ax, 6.6, 1.0, 1.9, 0.8, 'output $y(k)$\nlinear')
box(ax, 3.4, 0.05, 1.9, 0.6, 'context $u_c(k)=u(k-1)$')
arr(ax, (2.1, 1.4), (3.4, 1.4), '$w_2$'); arr(ax, (5.3, 1.4), (6.6, 1.4), '$w_3$')
arr(ax, (3.9, 0.65), (3.9, 1.0), None); arr(ax, (4.9, 1.0), (4.9, 0.65), None); ax.text(5.05, 0.78, '$w_1$', fontsize=7)
ax = axs[1]; ax.set_title('(b) QENN as implemented (6$\\times$5$\\times$1; quantum software only supplies the 2-vector of each hidden unit)', fontsize=8.5, loc='left')
box(ax, 0.1, 1.0, 1.5, 0.8, 'window\n$x(k)\\in\\mathbb{R}^6$'); box(ax, 2.1, 1.0, 2.1, 0.8, '5 hidden units\n$[\\cos e^{\\eta_i},\\ \\sin e^{\\eta_i}]$'); box(ax, 4.7, 1.0, 2.0, 0.8, 'tensor product\n$\\alpha(k)\\in\\mathbb{R}^{32}$'); box(ax, 7.2, 1.0, 2.6, 0.8, 'output unit\n$\\hat y=2\\sin^2(e^{\\eta_{out}})-1$')
box(ax, 4.7, 0.05, 2.0, 0.6, 'context $\\tilde a(k)$, gain $c=0.5$')
arr(ax, (1.6, 1.4), (2.1, 1.4)); arr(ax, (4.2, 1.4), (4.7, 1.4)); arr(ax, (6.7, 1.4), (7.2, 1.4))
arr(ax, (5.2, 1.0), (5.2, 0.65)); arr(ax, (4.7, 0.35), (3.15, 1.0), rad=-0.25)
plt.tight_layout(); plt.savefig('figs/fig1_architecture.pdf', bbox_inches='tight'); plt.savefig('figs/fig1_architecture.png', dpi=300, bbox_inches='tight'); plt.close()

# ---- Fig 2: series with train/test split (matched windows)
fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.8))
for ax, n in zip(axs.ravel(), D.FILES):
    d = D.prepare(n); x = np.arange(d['n_total'])
    ax.plot(x, d['raw'], lw=0.7, color='#1f77b4'); ax.axvline(d['n_train'] + 6, color='crimson', lw=0.9, ls='--')
    ax.axvspan(d['n_train'] + 6, d['n_total'], color='crimson', alpha=0.07)
    ax.set_title(f"{n}  ({d['train_start'][:4]}\u2013{d['test_end'][:4]})", fontsize=8); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
for ax in axs[1]: ax.set_xlabel('trading-day index', fontsize=7.5)
for ax in axs[:, 0]: ax.set_ylabel('close', fontsize=7.5)
plt.tight_layout(); plt.savefig('figs/fig2_series.pdf', bbox_inches='tight'); plt.savefig('figs/fig2_series.png', dpi=300, bbox_inches='tight'); plt.close()
print('ok')
