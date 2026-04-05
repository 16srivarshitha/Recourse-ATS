import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Canvas 
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(0, 100)
ax.set_ylim(5, 75)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

# Palette 
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_MLP   = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace'}

# Helpers 
def box(x, y, w, h, title, sub=None, theme=C_INPUT, title_size=10.5, sub_size=8.5):
    fc, ec, tc = theme
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.0,rounding_size=1.5',
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    if sub:
        ax.text(x, y + h * 0.14, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)
        ax.text(x, y - h * 0.22, sub, ha='center', va='center',
                fontsize=sub_size, color=tc, alpha=0.85, zorder=4, linespacing=1.4)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)

def arrow(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle=ls, mutation_scale=13), zorder=2)

def line(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=2)

# Title - Moved down to bridge the gap
ax.text(50, 68, "Architecture Component Diagram — Pure Semantic (Model 1)", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(50, 64.5, "Definitive reference · text-only cross-encoder pipeline",
        ha='center', va='center', fontsize=9, color='#777777')

# Visual box diagram 
shift_x = 5
base_y = 5 

# Inputs
box(27 + shift_x, 46 + base_y, 26, 7, "Resume text", "raw · up to 512 tokens", C_INPUT)
box(68 + shift_x, 46 + base_y, 26, 7, "Job description", "smart-parsed · 512 tokens", C_INPUT)

# Merge point label
ax.text(47.5 + shift_x, 39.5 + base_y, "[SEP]", ha='center', va='center',
        fontsize=8, color='#a39f96', **MONO)
line(27 + shift_x, 42.5 + base_y, 27 + shift_x, 39 + base_y)
line(68 + shift_x, 42.5 + base_y, 68 + shift_x, 39 + base_y)
line(27 + shift_x, 39 + base_y, 68 + shift_x, 39 + base_y)
arrow(47.5 + shift_x, 39 + base_y, 47.5 + shift_x, 35.5 + base_y)

# Cross-encoder
box(47.5 + shift_x, 32 + base_y, 38, 7, "Cross-Encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2 · 512 tokens", C_TEXT)

# CLS arrow + label
arrow(47.5 + shift_x, 28.5 + base_y, 47.5 + shift_x, 24.5 + base_y)
ax.text(50 + shift_x, 26.8 + base_y, "[CLS] 384-dim", ha='left', va='center',
        fontsize=8, color='#a39f96', **MONO)

# MLP
box(47.5 + shift_x, 21 + base_y, 38, 7, "MLP Head",
    "Linear(384->64) -> ReLU -> Linear(64->1) -> Sigmoid", C_MLP)

# Final score
arrow(47.5 + shift_x, 17.5 + base_y, 47.5 + shift_x, 13.5 + base_y)
box(47.5 + shift_x, 10 + base_y, 24, 7, "Match Score", "∈ [0, 1]", C_FINAL)

# Final Touches 
plt.tight_layout()
plt.savefig('architecture_diagram_pure_semantic.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_diagram_pure_semantic.png")
