import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

#  Canvas 
fig, ax = plt.subplots(figsize=(14, 9.5))
ax.set_xlim(0, 100)
ax.set_ylim(10, 102)
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
ax.text(50, 98, "Architecture Component Diagram — Cross-Attention (Model 3)", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(50, 94.5, "Token sequence queries skill graph · tight early fusion · skill info shapes text representation",
        ha='center', va='center', fontsize=9, color='#777777')

# LEFT PATHWAY: Text 
box(22, 85, 24, 7, "Resume text", "512 tokens", C_INPUT)
box(48, 85, 24, 7, "Job description", "512 tokens", C_INPUT)

line(22, 81.5, 22, 78)
line(48, 81.5, 48, 78)
line(22, 78, 48, 78)
arrow(35, 78, 35, 74.5)
ax.text(36.5, 76.5, "[SEP] concat", ha='left', va='center', fontsize=8, color=LINE, **MONO)

box(35, 71, 38, 7, "Cross-Encoder", "ms-marco-MiniLM-L-6-v2 · 512 tokens", C_TEXT)

# Full token sequence
arrow(35, 67.5, 35, 63.5)
box(35, 60, 36, 7, "Token sequence", "full sequence (B, 512, 384-dim)", C_TEXT, ts=9.5)

# RIGHT PATHWAY: Skills 
box(82, 85, 24, 7, "Resume + JD skills", "combined skill ID list", C_INPUT)
arrow(82, 81.5, 82, 77.5)
ax.text(83.5, 79.5, "GNN lookup", ha='left', va='center', fontsize=8, color='#196348', **MONO)

box(82, 74, 24, 7, "Skill embeddings", "variable seq · 128-dim", C_GRAPH)
arrow(82, 70.5, 82, 66.5)

# Projection
box(82, 63, 24, 7, "Linear projection", "Linear(128 -> 384)", C_GRAPH, ts=9.5)
arrow(82, 59.5, 82, 56)
ax.text(83.5, 57.5, "pad_sequence", ha='left', va='center', fontsize=8, color='#196348', **MONO)

# Cross-Attention Block 
# Route text sequence and skill projection into MHA
line(35, 56.5, 35, 51)
line(82, 56, 82, 51)
line(35, 51, 82, 51)

# Q/K/V labels
ax.text(32, 53, "Q", ha='center', va='center', fontsize=10, fontweight='bold', color='#7a3800', zorder=5)
ax.text(82, 52.5, "K, V", ha='center', va='center', fontsize=10, fontweight='bold', color='#7a3800', zorder=5)

arrow(58.5, 51, 58.5, 47.5)

box(58.5, 42.5, 52, 10, "Multi-Head Cross-Attention", 
    "Query = token seq · Key = Value = skill proj\nembed_dim=384 · num_heads=4 · batch_first=True", C_ATTN)

arrow(58.5, 37.5, 58.5, 33.5)
ax.text(60, 35.5, "attn_out (B, 512, 384)", ha='left', va='center', fontsize=8, color=LINE, **MONO)

# [CLS] Token Bypass & Concat 
# Cleanly route the CLS bypass out to the left to avoid crossing boxes
line(35, 56.5, 14, 56.5, ls='--', lw=1.0)
line(14, 56.5, 14, 31.5)
arrow(14, 31.5, 43, 31.5, ls='--', lw=1.0)

# CLS Badge
ax.add_patch(FancyBboxPatch((11.5, 43), 5, 3, boxstyle='round,pad=0.2', facecolor='#edeafc', edgecolor='#c4bdf0', zorder=3))
ax.text(14, 44.5, "[CLS]", ha='center', va='center', fontsize=8, color='#3d2685', **MONO, zorder=4)

ax.text(58.5, 31.5, "take attn_out[:, 0, :] · concat with cls_emb", ha='center', va='center', fontsize=8, color=LINE, **MONO)

arrow(58.5, 30.5, 58.5, 27)
ax.text(60, 28.5, "768-dim (384 + 384)", ha='left', va='center', fontsize=8, color=LINE, **MONO)

# Output & Score 
box(58.5, 23.5, 44, 7, "MLP Head", "Linear(768->64) -> ReLU -> Linear(64->1) -> Sigmoid", C_MLP)

arrow(58.5, 20, 58.5, 16.5)
box(58.5, 13, 26, 7, "Match Score", "∈ [0, 1]", C_FINAL)

# Save 
plt.tight_layout()
plt.savefig('architecture_diagram_cross_attn.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_diagram_cross_attn.png")
