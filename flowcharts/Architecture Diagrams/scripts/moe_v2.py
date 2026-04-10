import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 16))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

# Color Palettes
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_GRAPH = ('#eafcf2', '#bdf0d4', '#1f6b45')
C_GATE  = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace', 'fontsize': 8.5, 'color': '#666666'}

def box(x, y, w, h, title, text_lines=None, theme=C_INPUT, shape=None):
    # Draw background box
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.0,rounding_size=1', 
                                facecolor=theme[0], edgecolor=theme[1], linewidth=1.4, zorder=3))
    # Title
    ax.text(x, y + h/2 - 1.5, title, ha='center', va='top', fontsize=10.5, fontweight='bold', color=theme[2], zorder=4)
    
    # Layer details (left-aligned)
    if text_lines:
        content = "\n".join(text_lines)
        ax.text(x - w/2 + 1.5, y + h/2 - 3.5, content, ha='left', va='top', fontsize=9, color=theme[2], linespacing=1.6, zorder=4)
    
    # Tensor Shape tag (bottom right)
    if shape:
        ax.text(x + w/2 - 1.5, y - h/2 + 1, shape, ha='right', va='bottom', bbox=dict(facecolor='#ffffff', edgecolor=theme[1], boxstyle='round,pad=0.3', alpha=0.8), zorder=5, **MONO)

def ortho_path(points, color=LINE, lw=1.4, label=None):
    """Draws a path of strictly horizontal/vertical lines with an arrowhead at the end."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, zorder=2)
    # Add arrowhead at the exact end of the last segment
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), 
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=12), zorder=3)
    if label:
        # Place label on the first horizontal/vertical segment
        mid_x, mid_y = (xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8.5, color=color, 
                bbox=dict(facecolor='#f9f9f7', edgecolor='none', pad=2), zorder=5, fontfamily='monospace')

# Titles
ax.text(50, 97, "Neural Architecture & Dimensions — Mixture of Experts v2", ha='center', va='center', fontsize=16, fontweight='bold', color='#2a2a2a')
ax.text(50, 95, "Layer-by-layer parameter design and tensor shape transformations", ha='center', va='center', fontsize=10, color='#777777')

# =========================================================
# 1. Inputs
# =========================================================
box(25, 89, 32, 5, "Resume & JD Text Input", ["Tokenized input_ids & attention_mask"], C_INPUT, shape="[B, 256]")
box(75, 89, 32, 5, "Resume & JD Skill Sets", ["Extracted Vocabulary Node IDs"], C_INPUT, shape="[B, N]")

ortho_path([(25, 86.5), (25, 83)])
ortho_path([(75, 86.5), (75, 83)])

# =========================================================
# 2. Expert Encoders
# =========================================================
text_ops = [
    "1. all-MiniLM-L6-v2 Encoder",
    "2. Masked Mean Pooling",
    "3. Linear Projection (384 → 64)",
    "4. LayerNorm(64) + L2 Normalize"
]
box(25, 73, 32, 20, "Text Bi-Encoder Pipeline", text_ops, C_TEXT, shape="[B, 64]")

graph_ops = [
    "1. nn.Embedding(2502, 128)",
    "2. Self-Attn Neighbourhood Blend",
    "3. Inverse-Degree JD Weights",
    "4. Cross-Attention (Q=JD, K=V=Res)",
    "5. LayerNorm(128)",
    "6. Mean Pooling (over JD skills)"
]
box(75, 73, 32, 20, "GNN + Cross-Attention Pipeline", graph_ops, C_GRAPH, shape="[B, 128]")

ortho_path([(25, 63), (25, 56)])
ortho_path([(75, 63), (75, 56)])

# =========================================================
# 3. Scoring Heads
# =========================================================
box(25, 51, 32, 10, "Text Score Calculation", ["1. Cosine Similarity (Res · JD)", "2. Shift scale: (Sim + 1) / 2"], C_TEXT, shape="[B, 1]")
box(75, 51, 32, 10, "Graph MLP Score Head", ["1. Linear(128 → 32) + ReLU", "2. Linear(32 → 1)", "3. Sigmoid Activation"], C_GRAPH, shape="[B, 1]")

# Skill Count
box(50, 68, 14, 6, "Skill Count", ["Normed / 222"], C_INPUT, shape="[B, 1]")
ortho_path([(91, 89), (96, 89), (96, 68), (57, 68)]) # Routing from Input skills to count

# =========================================================
# 4. Gating Network
# =========================================================
# Routing to Gate Concat
ortho_path([(25, 46), (25, 40), (41, 40), (41, 37)])
ortho_path([(75, 46), (75, 40), (59, 40), (59, 37)])
ortho_path([(50, 65), (50, 37)])

box(50, 34, 40, 6, "Concat Features", ["torch.cat([T_Score, G_Score, Count], dim=1)"], C_INPUT, shape="[B, 3]")

ortho_path([(50, 31), (50, 28)])

gate_ops = [
    "1. Linear(3 → 32) + ReLU",
    "2. Linear(32 → 2)",
    "3. Softmax(dim=1)",
    "4. Floor Clamp (min=0.15) & Renormalize"
]
box(50, 20, 32, 16, "Gating Network", gate_ops, C_GATE, shape="[B, 2]")

# =========================================================
# 5. Output Assembly
# =========================================================
# Routing to Final Assembly
ortho_path([(50, 12), (50, 8)])
# Bypass text/graph scores down to assembly (using outside channels)
ortho_path([(9, 51), (4, 51), (4, 5), (14, 5)], label="Text Score")
ortho_path([(91, 51), (96, 51), (96, 5), (86, 5)], label="Graph Score")

box(50, 5, 72, 6, "Output Assembly", ["Final Score = (Gate_T * Text_Score) + (Gate_G * Graph_Score)"], C_FINAL, shape="[B, 1]")

plt.tight_layout()
plt.savefig('architecture_diagram_moev2.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_diagram_moev2.png")