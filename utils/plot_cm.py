import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cm(cm0, cm_type, level, save_dir):
    
    if cm_type == 'norm':
        fmt = ''
    elif cm_type == 'raw':
        fmt = 'd'   

    ax = sn.heatmap(
                    cm0,
                    annot=True,
                    cbar=True,
                    cbar_kws={'ticks': [-0.1]},
                    annot_kws={'size': 26, 'fontweight': 'bold'},
                    cmap='Blues',
                    fmt=fmt,
                    linewidths=0.5
                    )

    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=2, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=2, color='k', linewidth=4)

    ax.tick_params(direction='out', length=4, width=2, colors='k')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.tight_layout()
    
    fn = 'cm' + '_' + str(cm_type) + '_' + str(level) + '.png'
    plt.savefig(
                os.path.join(save_dir, fn),
                format='png',
                dpi=600
                )
    plt.close()
