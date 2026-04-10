import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 115) 
ax.set_ylim(15, 105)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_GRAPH = ('#e6f5f0', '#aedbc9', '#196348')
C_POOL  = ('#e8f7ef', '#5cb88a', '#0e4f35') 
C_MLP   = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace'}

def box(x, y, w, h, title, sub=None, theme=C_INPUT, ts=10.5, ss=8.5):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.0,rounding_size=1.5', facecolor=theme[0], edgecolor=theme[1], linewidth=1.4, zorder=3))
    if sub:
        ax.text(x, y + h * 0.14, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=theme[2], zorder=4)
        ax.text(x, y - h * 0.22, sub, ha='center', va='center', fontsize=ss, color=theme[2], alpha=0.85, zorder=4, linespacing=1.3)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=theme[2], zorder=4)

def arrow(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=ls, mutation_scale=13), zorder=2)

ax.text(57.5, 101, "Architecture Component Diagram — Late Fusion (Model 2)", ha='center', va='center', fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(57.5, 97.5, "Text represents Candidate · Graph evaluates Candidate vs JD intersection", ha='center', va='center', fontsize=9, color='#777777')

box(26, 85, 24, 7, "Resume text", "raw text", C_INPUT)
arrow(26, 81.5, 26, 75.5)
box(26, 72, 38, 7, "Text Encoder", "all-MiniLM-L6-v2", C_TEXT)
arrow(26, 68.5, 26, 52.5)
box(26, 49, 26, 7, "Resume Embedding", "Mean Pool (384-dim)", C_TEXT)

box(76, 85, 20, 7, "Resume skills", "skill ID list", C_INPUT)
box(100, 85, 20, 7, "JD skills", "skill ID list", C_INPUT)
arrow(76, 81.5, 76, 78.5)
arrow(100, 81.5, 100, 78.5)

box(76, 75, 20, 7, "Skill Embeddings", "GNN lookup (128-dim)", C_GRAPH, ts=9.5, ss=7.5)
box(100, 75, 20, 7, "Skill Embeddings", "GNN lookup (128-dim)", C_GRAPH, ts=9.5, ss=7.5)

arrow(76, 71.5, 76, 65.5)
ax.plot([100, 100], [71.5, 68], color=LINE, lw=1.4, zorder=2)
ax.plot([88, 104], [68, 68], color=LINE, lw=1.4, zorder=2)
arrow(88, 68, 88, 65.5) 
arrow(104, 68, 104, 64.5) 

box(78, 61, 30, 9, "JD-Attended Skill Pooler", "JD=Query · Res=Key/Value\nMultiheadAttention + LayerNorm", C_POOL, ts=9.5, ss=7.5)
box(104, 61, 16, 7, "JD Mean", "128-dim", C_GRAPH, ts=9.5)

arrow(78, 56.5, 78, 52.5)
arrow(104, 57.5, 104, 52.5)

box(89, 49, 36, 7, "Graph Representation", "concat: res_attn(128) + jd_mean(128)\n= 256-dim", C_GRAPH, ts=9.5, ss=8)

ax.plot([26, 26], [45.5, 42], color=LINE, lw=1.4, zorder=2)
ax.plot([89, 89], [45.5, 42], color=LINE, lw=1.4, zorder=2)
ax.plot([26, 89], [42, 42], color=LINE, lw=1.4, zorder=2)
arrow(57.5, 42, 57.5, 37.5)
ax.text(59, 40, "concat: 384 + 256 = 640-dim", ha='left', va='center', fontsize=8, color=LINE, **MONO)

box(57.5, 34, 42, 7, "MLP Head", "Linear(640->128) -> ReLU -> Linear(128->1) -> Sigmoid", C_MLP)

arrow(57.5, 30.5, 57.5, 26.5)
box(57.5, 23, 26, 7, "Match Score", "∈ [0, 1]", C_FINAL)

plt.tight_layout()
plt.savefig('architecture_diagram_late_fusion.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_diagram_late_fusion.png")
