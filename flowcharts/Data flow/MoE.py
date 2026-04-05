import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

#  Canvas 
fig, ax = plt.subplots(figsize=(12, 11.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 110)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

#  Palette 
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_GRAPH = ('#e6f5f0', '#aedbc9', '#196348')
C_GATE  = ('#fcefe6', '#f0cdb1', '#8c3620')
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
ax.text(50, 105, "Conceptual Data Flow — Mixture of Experts (Model 4)", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(50, 101.5, "Independent pathways evaluated by specialized experts and dynamically weighted",
        ha='center', va='center', fontsize=9, color='#777777')

#  Flow Diagram 
b_w = 38
b_h = 8

# Level 1: Inputs
box(30, 92, b_w, b_h, "1a. Text Inputs", "Resume & JD Text", C_INPUT)
box(70, 92, b_w, b_h, "1b. Skill Inputs", "Resume & JD Skill IDs", C_INPUT)

arrow(30, 88, 30, 81)
arrow(70, 88, 70, 81)

# Level 2: Encoders
box(30, 77, b_w, b_h, "2a. Sequence Encoding", "Cross-Encoder -> Extract [CLS]", C_TEXT)
box(70, 77, b_w, b_h, "2b. Graph Encoding", "GNN Lookup -> Poolers -> Concat", C_GRAPH)

arrow(30, 73, 30, 66)
arrow(70, 73, 70, 66)

# Level 3: Experts
box(30, 62, b_w, b_h, "3a. Text Expert MLP", "Maps text emb to specialized score", C_TEXT)
box(70, 62, b_w, b_h, "3b. Graph Expert MLP", "Maps graph emb to specialized score", C_GRAPH)

arrow(30, 58, 30, 51)
arrow(70, 58, 70, 51)

# Level 4: Independent Scores
box(30, 47, b_w, b_h, "4a. Text Score", "Scalar ∈ [0, 1]", C_TEXT)
box(70, 47, b_w, b_h, "4b. Graph Score", "Scalar ∈ [0, 1]", C_GRAPH)

# Convergence to Gate
arrow(30, 43, 42, 34)
arrow(70, 43, 58, 34)

# Level 5: Gating Network
box(50, 30, 50, b_h, "5. Gating Network", "Evaluates ([CLS] + Text Score + Graph Score) -> Weights", C_GATE)

arrow(50, 26, 50, 17)

# Level 6: Output
box(50, 13, 50, b_h, "6. Final Match Score", "Sum: (w_text × text_score) + (w_graph × graph_score)", C_FINAL)

#  Save 
plt.tight_layout()
plt.savefig('arch_4_conceptual_flow.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> arch_4_conceptual_flow.png")
