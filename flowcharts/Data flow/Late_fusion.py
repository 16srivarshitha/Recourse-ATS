import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

#  Canvas 
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

#  Palette 
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_GRAPH = ('#e6f5f0', '#aedbc9', '#196348')
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
ax.text(50, 96, "Conceptual Data Flow — Late Fusion (Model 2)", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(50, 92.5, "Parallel text and graph pathways converging at the MLP head",
        ha='center', va='center', fontsize=9, color='#777777')

#  Flow Diagram 
b_w = 36
b_h = 8

# Level 1: Inputs (Two columns)
box(30, 80, b_w, b_h, "1a. Text Inputs", "Resume & JD Text", C_INPUT)
box(70, 80, b_w, b_h, "1b. Skill Inputs", "Resume & JD Skill IDs", C_INPUT)

arrow(30, 76, 30, 67)
arrow(70, 76, 70, 67)

# Level 2: Processing
box(30, 63, b_w, b_h, "2a. Sequence Formatting", "Concat: [Resume] [SEP] [JD]", C_TEXT)
box(70, 63, b_w, b_h, "2b. GNN Lookups", "Extract 128-dim graph embeddings", C_GRAPH)

arrow(30, 59, 30, 50)
arrow(70, 59, 70, 50)

# Level 3: Embeddings / Pooling
box(30, 46, b_w, b_h, "3a. Cross-Encoder", "[CLS] Token Embedding (384-dim)", C_TEXT)
box(70, 46, b_w, b_h, "3b. Skill Pooling", "Attended (Res) + Mean (JD) = 256-dim", C_GRAPH)

# Convergence arrows
arrow(30, 42, 45, 33)
arrow(70, 42, 55, 33)

# Level 4: Late Fusion
box(50, 29, 44, b_h, "4. Late Fusion (Concatenation)", "Text (384) + Graph (256) = 640-dim", C_MLP)

arrow(50, 25, 50, 16)

# Level 5: Output
box(50, 12, 30, b_h, "5. Match Score", "MLP -> Scalar ∈ [0, 1]", C_FINAL)

#  Save 
plt.tight_layout()
plt.savefig('arch_2_conceptual_flow.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> arch_2_conceptual_flow.png")
