import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Canvas 
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, 120)
ax.set_ylim(0, 105)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

# Palette 
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_GRAPH = ('#e6f5f0', '#aedbc9', '#196348')
C_GATE  = ('#fcefe6', '#f0cdb1', '#8c3620')
C_POOL  = ('#e8f7ef', '#5cb88a', '#0e4f35')
C_MLP   = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace'}

# Helpers 
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

# Title
ax.text(60, 102, "Architecture Component Diagram — Mixture of Experts (Model 4)", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(60, 98.5, "Independent Experts: Text scores candidate | Graph scores JD fit | Gate dynamically combines per-pair",
        ha='center', va='center', fontsize=9, color='#777777')


# ==========================================
# LEFT PATHWAY: TEXT (Centered at X=28)
# ==========================================
box(28, 91, 24, 7, "Resume text", "raw text", C_INPUT)
arrow(28, 87.5, 28, 82.5)

box(28, 79, 36, 7, "Text Encoder", "all-MiniLM-L6-v2", C_TEXT)
arrow(28, 75.5, 28, 70.5)

box(28, 67, 30, 7, "Resume Emb (384)", "Mean Pooled", C_TEXT)
arrow(28, 63.5, 28, 56.5)

box(28, 53, 30, 7, "Text Expert MLP", "Linear(384->64) -> ReLU -> ...", C_TEXT, ts=9.5, ss=8)
arrow(28, 49.5, 28, 43.5)

box(28, 40, 24, 7, "t_score", "Candidate fit ∈ [0,1]", C_TEXT)


# ==========================================
# RIGHT PATHWAY: GRAPH (Centered around X=92)
# ==========================================
box(78, 91, 22, 7, "Resume skills", "skill ID list", C_INPUT)
box(106, 91, 22, 7, "JD skills", "skill ID list", C_INPUT)

arrow(78, 87.5, 78, 82.5)
arrow(106, 87.5, 106, 82.5)

box(78, 79, 22, 7, "Skill Embeddings", "GNN (128-dim)", C_GRAPH, ts=9.5, ss=8)
box(106, 79, 22, 7, "Skill Embeddings", "GNN (128-dim)", C_GRAPH, ts=9.5, ss=8)

# Routing into Poolers
ax.plot([106, 106], [75.5, 73.5], color=LINE, lw=1.4, zorder=2)
ax.plot([94, 108], [73.5, 73.5], color=LINE, lw=1.4, zorder=2)
arrow(94, 73.5, 94, 71.5)
arrow(108, 73.5, 108, 70.5)
arrow(78, 75.5, 78, 71.5)

box(82, 67, 32, 9, "JD-Attended Skill Pooler", "MHA(128) + LayerNorm", C_POOL, ts=9.5, ss=8)
box(108, 67, 16, 7, "JD Mean", "128-dim", C_GRAPH, ts=9.5)

# Concat
ax.plot([82, 82], [62.5, 60], color=LINE, lw=1.4, zorder=2)
ax.plot([108, 108], [63.5, 60], color=LINE, lw=1.4, zorder=2)
ax.plot([82, 108], [60, 60], color=LINE, lw=1.4, zorder=2)
arrow(92, 60, 92, 56.5)
ax.text(93.5, 58.2, "concat -> 256-dim", ha='left', va='center', fontsize=8, color=LINE, **MONO)

box(92, 53, 32, 7, "Graph Expert MLP", "Linear(256->64) -> ReLU -> ...", C_GRAPH, ts=9.5, ss=8)
arrow(92, 49.5, 92, 43.5)

box(92, 40, 24, 7, "g_score", "Graph match ∈ [0,1]", C_GRAPH)


# ==========================================
# GATING NETWORK & BYPASSES
# ==========================================
# Left Bypass (Text)
ax.plot([28, 10], [60.5, 60.5], ls='--', lw=1.4, color=LINE, zorder=2)
ax.plot([10, 10], [60.5, 31], ls='--', lw=1.4, color=LINE, zorder=2)
ax.plot([10, 45], [31, 31], ls='--', lw=1.4, color=LINE, zorder=2)
arrow(45, 31, 45, 28.5, ls='--', lw=1.4)

ax.add_patch(FancyBboxPatch((7, 44), 6, 3, boxstyle='round,pad=0.2', facecolor='#edeafc', edgecolor='#c4bdf0', zorder=3))
ax.text(10, 45.5, "Text Emb bypass", ha='center', va='center', fontsize=7.5, color='#3d2685', **MONO, zorder=4)

# Right Bypass (Graph)
ax.plot([92, 112], [58.2, 58.2], ls='--', lw=1.4, color=LINE, zorder=2)
ax.plot([112, 112], [58.2, 31], ls='--', lw=1.4, color=LINE, zorder=2)
ax.plot([112, 75], [31, 31], ls='--', lw=1.4, color=LINE, zorder=2)
arrow(75, 31, 75, 28.5, ls='--', lw=1.4)

ax.add_patch(FancyBboxPatch((105.5, 44), 6.5, 3, boxstyle='round,pad=0.2', facecolor='#e6f5f0', edgecolor='#aedbc9', zorder=3))
ax.text(108.75, 45.5, "Graph Emb bypass", ha='center', va='center', fontsize=7.5, color='#196348', **MONO, zorder=4)

# Gating Box
box(60, 24, 48, 9, "Gating Network", "Input: Concat(Text Emb, Graph Emb)\nOutputs (w_text, w_graph)", C_GATE)


# ==========================================
# FINAL EQUATION & OUTPUT
# ==========================================
# Route scores down to the equation
ax.plot([28, 28], [36.5, 14], color=LINE, lw=1.4, ls='--', zorder=2)
arrow(28, 14, 32, 14, ls='--', lw=1.4)

ax.plot([92, 92], [36.5, 14], color=LINE, lw=1.4, ls='--', zorder=2)
arrow(92, 14, 88, 14, ls='--', lw=1.4)

# Gate output to equation
arrow(60, 19.5, 60, 16.5)

# Equation Box (Done as a standard Box for cleanliness)
box(60, 14, 56, 5, "w_text × t_score  +  w_graph × g_score", theme=C_FINAL, ts=11)

# Final output
arrow(60, 11.5, 60, 8.5)
box(60, 5, 28, 7, "Final Match Score", "∈ [0, 1]", C_FINAL)

# Save
plt.tight_layout()
plt.savefig('architecture_diagram_moe.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_diagram_moe.png")
