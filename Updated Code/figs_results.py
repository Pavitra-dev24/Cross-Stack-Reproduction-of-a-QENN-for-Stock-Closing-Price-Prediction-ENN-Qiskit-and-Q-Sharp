import sys; sys.path.insert(0,'/home/claude/work')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import data as D
from baselines import persistence
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
R='/home/claude/work/results/'; ORDER=["BSE","NASDAQ","HSI","SSE","Russell2000","TAIEX"]
def load(f): return pd.read_csv(R+f).drop_duplicates(['dataset','scaling','model','seed'])
ALL=pd.concat([load('stage1.csv'),load('stage2.csv'),load('stage3.csv')],ignore_index=True)
# ---- Fig 3: NMSE bars (log scale), train-only scaling
models=[("Persistence","Persistence"),("ARIMA","ARIMA"),("ENN-12 (tuned)","Elman-12"),("GRU-6 (tuned)","GRU-6"),("LSTM-5 (tuned)","LSTM-5"),
        ("QENN-B DCQGA (corrected fitness)","QENN-B DCQGA"),("QENN-O DCQGA (corrected fitness)","QENN-O DCQGA"),("QENN-B (tuned)","QENN-B SGD")]
cols=['#4c4c4c','#8c8c8c','#1f77b4','#2ca02c','#ff7f0e','#d62728','#9467bd','#e377c2']
fig,ax=plt.subplots(figsize=(7.2,2.9)); w=0.8/len(models); x=np.arange(len(ORDER))
for i,(m,lab) in enumerate(models):
    v=[ALL[(ALL.model==m)&(ALL.dataset==d)&(ALL.scaling=='train')].nmse.mean() for d in ORDER]
    e=[ALL[(ALL.model==m)&(ALL.dataset==d)&(ALL.scaling=='train')].nmse.std(ddof=1) for d in ORDER]
    ax.bar(x+(i-len(models)/2)*w+w/2,v,width=w,label=lab,color=cols[i])
fl=pd.read_csv(R+'qenn_range_floor.csv').floor_nmse.values
for xi,fv in zip(x,fl):
    if fv>0: ax.hlines(fv,xi-0.4,xi+0.4,colors='k',linestyles=':',lw=1.0)
ax.plot([],[],'k:',lw=1,label='QENN output-range floor')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(["BSE","NASDAQ","HSI","SSE","Russell 2000","TAIEX"]); ax.set_ylabel('test NMSE (log)')
ax.grid(axis='y',alpha=0.3); ax.legend(ncol=3,fontsize=6.5,loc='upper left',frameon=False,bbox_to_anchor=(0,1.30))
plt.tight_layout(); plt.savefig('figs/fig3_nmse.pdf',bbox_inches='tight'); plt.savefig('figs/fig3_nmse.png',dpi=300,bbox_inches='tight'); plt.close()
# ---- Fig 4: BSE predictions
d=D.prepare('BSE'); y=d['yte']
P={'Persistence':persistence(d),'GRU-6 (tuned)':np.load(R+'pred_BSE_GRU-6_tuned.npy'),'QENN-B, DCQGA':np.load(R+'pred_BSE_QENN-B_DCQGA_corrected_fitness.npy')}
fig,axs=plt.subplots(3,1,figsize=(7.2,4.4),sharex=True)
for ax,(k,p) in zip(axs,P.items()):
    ax.plot(y,color='#1f77b4',lw=0.9,label='actual'); ax.plot(p,color='crimson',lw=0.9,ls='--',label=k); ax.axhline(1,color='k',lw=0.6,ls=':')
    ax.set_ylabel('scaled close',fontsize=7.5); ax.legend(fontsize=7,loc='upper left',frameon=False); ax.grid(alpha=0.25)
axs[-1].set_xlabel('BSE test step (dotted line: upper edge of the training range, $+1$)',fontsize=7.5)
plt.tight_layout(); plt.savefig('figs/fig4_bse_pred.pdf',bbox_inches='tight'); plt.savefig('figs/fig4_bse_pred.png',dpi=300,bbox_inches='tight'); plt.close()
# ---- Fig 5: learning-rate trajectories, released vs corrected fitness (BSE, seed 0)
import glob
fig,axs=plt.subplots(1,2,figsize=(7.2,2.4),sharey=True)
for ax,(fn,t) in zip(axs,[('lrhist_BSE_QENN-B_released.npy','released fitness (pre-update loss)'),('lrhist_BSE_QENN-B_corrected.npy','corrected fitness (post-update loss)')]):
    try:
        h=np.load(R+fn)
        for j,lab in enumerate(['$\\eta_1$','$\\eta_2$','$\\eta_3$']): ax.plot(h[:,j],lw=1.1,label=lab)
    except Exception as ex: print('missing',fn,ex)
    ax.set_title(t,fontsize=8); ax.set_xlabel('epoch',fontsize=7.5); ax.grid(alpha=0.25)
axs[0].set_ylabel('learning rate'); axs[0].legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig('figs/fig5_lr.pdf',bbox_inches='tight'); plt.savefig('figs/fig5_lr.png',dpi=300,bbox_inches='tight'); plt.close()
print('figs ok')
