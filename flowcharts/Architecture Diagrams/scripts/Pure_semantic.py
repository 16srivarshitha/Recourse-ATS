import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 100)
ax.set_ylim(5, 85)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_MLP   = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace'}

def box(x, y, w, h, title, sub=None, theme=C_INPUT, ts=10.5, ss=8.5):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.0,rounding_size=1.5', facecolor=theme[0], edgecolor=theme[1], linewidth=1.4, zorder=3))
    if sub:
        ax.text(x, y + h * 0.14, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=theme[2], zorder=4)
        ax.text(x, y - h * 0.22, sub, ha='center', va='center', fontsize=ss, color=theme[2], alpha=0.85, zorder=4, linespacing=1.4)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=theme[2], zorder=4)

def arrow(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=ls, mutation_scale=13), zorder=2)

ax.text(50, 80, "Architecture Component Diagram — Pure Semantic (Model 1)", ha='center', va='center', fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(50, 76.5, "Bi-Encoder (Twin Tower) pipeline · text semantics only", ha='center', va='center', fontsize=9, color='#777777')

box(28, 65, 26, 7, "Resume text", "raw · up to 256 tokens", C_INPUT)
box(72, 65, 26, 7, "Job description", "smart-parsed · 256 tokens", C_INPUT)

arrow(28, 61.5, 28, 55.5)
arrow(72, 61.5, 72, 55.5)

box(28, 52, 32, 7, "Text Encoder", "all-MiniLM-L6-v2 (Frozen/Fine-tuned)", C_TEXT)
box(72, 52, 32, 7, "Text Encoder", "all-MiniLM-L6-v2 (Frozen/Fine-tuned)", C_TEXT)

arrow(28, 48.5, 28, 42.5)
arrow(72, 48.5, 72, 42.5)

box(28, 39, 24, 7, "Resume Embedding", "Mean Pooled [384-dim]", C_TEXT)
box(72, 39, 24, 7, "JD Embedding", "Mean Pooled [384-dim]", C_TEXT)

ax.plot([28, 43], [35.5, 30], color=LINE, lw=1.4, zorder=2)
ax.plot([72, 57], [35.5, 30], color=LINE, lw=1.4, zorder=2)
ax.annotate('', xy=(50, 28), xytext=(43, 30), arrowprops=dict(arrowstyle='-', color=LINE, lw=1.4), zorder=2)
ax.annotate('', xy=(50, 28), xytext=(57, 30), arrowprops=dict(arrowstyle='-', color=LINE, lw=1.4), zorder=2)

box(50, 24, 28, 7, "Cosine Similarity", "((Res * JD).sum() + 1) / 2", C_MLP)

arrow(50, 20.5, 50, 16)
box(50, 12, 42, 8, "MLP Head & Match Score", "Linear(1->64) -> ReLU -> Linear(64->1) -> Sigmoid", C_FINAL)

plt.tight_layout()
plt.savefig('architecture_diagram_pure_semantic.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_diagram_pure_semantic.png")
