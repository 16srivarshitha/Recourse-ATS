import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

#  Canvas 
fig, ax = plt.subplots(figsize=(15, 12))
ax.set_xlim(0, 120)
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
C_POOL  = ('#e8f7ef', '#5cb88a', '#0e4f35')
C_MLP   = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace'}

#  Helpers 
def box(x, y, w, h, title, sub=None, theme=C_INPUT, ts=10.5, ss=8.5):
    fc, ec, tc = theme
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.0,rounding_size=1.5',
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    if sub:
        ax.text(x, y + h * 0.14, title, ha='center', va='center',
                fontsize=ts, fontweight='bold', color=tc, zorder=4)
        ax.text(x, y - h * 0.22, sub, ha='center', va='center',
                fontsize=ss, color=tc, alpha=0.85, zorder=4, linespacing=1.4)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=ts, fontweight='bold', color=tc, zorder=4)

def arrow(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle=ls, mutation_scale=13), zorder=2)

def line(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=2)

#  Title ─
ax.text(60, 105, "Architecture Component Diagram — Mixture of Experts (Model 4)", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(60, 101.5, "Two specialized experts kept separate throughout · learned gate combines per-pair",
        ha='center', va='center', fontsize=9, color='#777777')

#  TEXT PATHWAY (Left side, centered at X=33) 
box(20, 89, 22, 7, "Resume text", "512 tokens", C_INPUT)
box(46, 89, 22, 7, "Job description", "512 tokens", C_INPUT)

line(20, 85.5, 20, 82)
line(46, 85.5, 46, 82)
line(20, 82, 46, 82)
arrow(33, 82, 33, 78.5)
ax.text(34.5, 80, "[SEP] concat", ha='left', va='center', fontsize=8, color=LINE, **MONO)

box(33, 75, 36, 7, "Cross-Encoder", "ms-marco-MiniLM-L-6-v2 · 512 tokens", C_TEXT)

arrow(33, 71.5, 33, 59.5)
ax.text(34.5, 65.5, "[CLS] 384-dim", ha='left', va='center', fontsize=8, color=LINE, **MONO)

# Text Expert aligned with Graph Expert
box(33, 56, 30, 7, "Text Expert MLP", "Linear(384->64) -> ReLU -> ...", C_TEXT, ts=9.5, ss=8)

arrow(33, 52.5, 33, 45.5)
ax.text(34.5, 49, "text_score ∈ [0,1]", ha='left', va='center', fontsize=8, color=LINE, **MONO)

box(33, 42, 22, 7, "text_score", "semantic fit", C_TEXT)


#  GRAPH PATHWAY (Right side, centered at X=87) 
box(74, 89, 22, 7, "Resume skills", "skill ID list", C_INPUT)
box(100, 89, 22, 7, "JD skills", "skill ID list", C_INPUT)

arrow(74, 85.5, 74, 81.5)
arrow(100, 85.5, 100, 81.5)

box(74, 78, 20, 7, "Skill Embeddings", "GNN (128-dim)", C_GRAPH, ts=9.5, ss=8)
box(100, 78, 20, 7, "Skill Embeddings", "GNN (128-dim)", C_GRAPH, ts=9.5, ss=8)

# JD Embs route into Attended Pooler (as Query) and Mean Pooler
line(100, 74.5, 100, 72)
line(88, 72, 106, 72)
arrow(88, 72, 88, 70.5)
arrow(106, 72, 106, 69.5)

# Res Embs route directly to Attended Pooler (as Key/Value)
arrow(74, 74.5, 74, 70.5)

# Poolers
box(78, 66, 30, 9, "JD-Attended Skill Pooler", "MHA(128) + LayerNorm", C_POOL, ts=9.5, ss=8)
box(106, 66, 14, 7, "JD Mean", "128-dim", C_GRAPH, ts=9.5)

# Concat
line(78, 61.5, 78, 59)
line(106, 62.5, 106, 59)
line(78, 59, 106, 59)
arrow(87, 59, 87, 59.5) # Wait arrow pointing down
arrow(87, 59, 87, 57.5) # Removed extra arrow
ax.text(88.5, 58.5, "concat -> 256-dim", ha='left', va='center', fontsize=8, color=LINE, **MONO)

# Graph Expert
box(87, 56, 30, 7, "Graph Expert MLP", "Linear(256->64) -> ReLU -> ...", C_GRAPH, ts=9.5, ss=8)

arrow(87, 52.5, 87, 45.5)
ax.text(88.5, 49, "graph_score ∈ [0,1]", ha='left', va='center', fontsize=8, color=LINE, **MONO)

box(87, 42, 22, 7, "graph_score", "skill coverage", C_GRAPH)


#  GATING NETWORK & BYPASSES (Center) 
# [CLS] clean outer bypass loop
line(33, 65.5, 10, 65.5, ls='--', lw=1.0)
line(10, 65.5, 10, 31)
line(10, 31, 46, 31, ls='--', lw=1.0)
arrow(46, 31, 46, 28)

ax.add_patch(FancyBboxPatch((7, 46.5), 6, 3, boxstyle='round,pad=0.2', facecolor='#edeafc', edgecolor='#c4bdf0', zorder=3))
ax.text(10, 48, "[CLS] bypass", ha='center', va='center', fontsize=7.5, color='#3d2685', **MONO, zorder=4)

# Routing text_score and graph_score to Gate
line(33, 38.5, 33, 34)
line(33, 34, 55, 34)
arrow(55, 34, 55, 28)

line(87, 38.5, 87, 34)
line(87, 34, 65, 34)
arrow(65, 34, 65, 28)

ax.text(60, 31.5, "Gate Input: [CLS] + text_score + graph_score", ha='center', va='center', fontsize=8, color=LINE, **MONO)

# Gate Box
box(60, 23, 44, 10, "Gating Network", 
    "Linear(3->16) -> ReLU -> Linear(16->2) -> Softmax\nOutputs (w_text, w_graph)", C_GATE)

# Routing text_score and graph_score bypassing down to the final equation
line(33, 34, 33, 16)
arrow(33, 16, 45, 16, ls='--', lw=1.0)

line(87, 34, 87, 16)
arrow(87, 16, 75, 16, ls='--', lw=1.0)

# Gate output to equation
arrow(60, 18, 60, 13)

# Equation text
ax.add_patch(FancyBboxPatch((35, 13), 50, 4.5, boxstyle='round,pad=0.0,rounding_size=0.5', facecolor='#fcf1db', edgecolor='#e8cc97', alpha=0.3, zorder=1))
ax.text(60, 15.2, "w_text × text_score  +  w_graph × graph_score", ha='center', va='center', fontsize=9, fontweight='bold', color='#735012', **MONO, zorder=5)

# Final Output
box(60, 7, 26, 7, "Final Match Score", "∈ [0, 1]", C_FINAL)

#  Save 
plt.tight_layout()
plt.savefig('architecture_diagram_moe.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_diagram_moe.png")
