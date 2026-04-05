import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

#  Canvas 
fig, ax = plt.subplots(figsize=(12, 10.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 105)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

#  Palette 
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_GRAPH = ('#e6f5f0', '#aedbc9', '#196348')
C_ATTN  = ('#fff0e6', '#f0c49b', '#7a3800')
C_MLP   = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace'}

#  Helpers 
def box(x, y, w, h, title, sub=None, theme=C_INPUT, title_size=10.5, sub_size=8.5, linespacing=1.2):
    fc, ec, tc = theme
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.0,rounding_size=1.5',
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    if sub:
        ax.text(x, y + h * 0.15, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)
        ax.text(x, y - h * 0.22, sub, ha='center', va='center',
                fontsize=sub_size, color=tc, alpha=0.85, zorder=4, linespacing=linespacing)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)

def arrow(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE, label=None, label_size=8, label_pos=0.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle=ls, mutation_scale=13), zorder=2)
    if label:
        lx = x1 + (x2 - x1) * label_pos
        ly = y1 + (y2 - y1) * label_pos
        ax.text(lx, ly, label, ha='center', va='center', fontsize=label_size, color=color, zorder=5, **MONO, bbox=dict(facecolor='#f9f9f7', edgecolor='none', pad=1))

#  Title 
ax.text(50, 100, "Conceptual Data Flow — Cross-Attention (Model 3)", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(50, 96.5, "Skill graph actively queried by token sequence during encoding phase",
        ha='center', va='center', fontsize=9, color='#777777')

#  Flow Diagram 
b_w = 38
b_h = 8

# Level 1: Inputs
box(30, 85, b_w, b_h, "1a. Text Input", "Resume & JD Text", C_INPUT)
box(70, 85, b_w, b_h, "1b. Skill Input", "Combined Skills (Res + JD)", C_INPUT)

arrow(30, 81, 30, 72)
arrow(70, 81, 70, 72)

# Level 2: Processing Base
box(30, 68, b_w, b_h, "2a. Sequence Encoding", "Cross-Encoder full sequence extraction", C_TEXT)
box(70, 68, b_w, b_h, "2b. Graph Lookup", "Extract 128-dim GNN embeddings", C_GRAPH)

arrow(30, 64, 30, 55)
arrow(70, 64, 70, 55)

# Level 3: Dimensionality & Prep
box(30, 51, b_w, b_h, "3a. Token Sequence (Query)", "Tokens (B, 512, 384-dim)", C_TEXT)
box(70, 51, b_w, b_h, "3b. Skill Projection (Key/Value)", "Linear map (128->384-dim)", C_GRAPH)

# Convergence to Attention
arrow(30, 47, 45, 38)
arrow(70, 47, 55, 38)

# Level 4: Cross-Attention
box(50, 34, 48, b_h, "4. Multi-Head Cross-Attention", "Text actively shaped by skill embeddings", C_ATTN)

arrow(50, 30, 50, 21)

# Level 5: Fusion & Output
box(50, 17, 48, b_h, "5. Fusion & Prediction", "Concat Attn_Out[0] + [CLS] -> MLP -> Scalar", C_MLP)

#  Save 
plt.tight_layout()
plt.savefig('arch_3_conceptual_flow.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> arch_3_conceptual_flow.png")
