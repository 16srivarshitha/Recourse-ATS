import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Canvas Setup
fig, ax = plt.subplots(figsize=(10, 15)) # Taller for a vertical architecture stack
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

# Custom Color Palettes
C_INPUT = ('#eaf2f8', '#b5d0e8', '#28547a') # Soft Blue (Inputs)
C_GAT   = ('#edeafc', '#c4bdf0', '#3d2685') # Purple (Attention Layers)
C_NORM  = ('#f4f3ee', '#dcd8c8', '#4d4d4d') # Grey (Norm / Activations)
C_LOSS  = ('#fcedec', '#ebb0ad', '#7a221f') # Coral/Red (Loss Functions)
LINE    = '#a39f96'
MONO    = {'fontfamily': 'monospace'}

# Helpers
def box(x, y, w, h, title, sub=None, theme=C_INPUT, title_size=11, sub_size=9, linespacing=1.4):
    fc, ec, tc = theme
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.0,rounding_size=1.5',
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    if sub:
        ax.text(x, y + h * 0.18, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)
        ax.text(x, y - h * 0.15, sub, ha='center', va='center',
                fontsize=sub_size, color=tc, alpha=0.9, zorder=4, linespacing=linespacing)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)

def arrow(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE, label=None, label_size=9, label_pos=0.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle=ls, mutation_scale=15), zorder=2)
    if label:
        lx = x1 + (x2 - x1) * label_pos
        ly = y1 + (y2 - y1) * label_pos
        # Add background block to label for readability over lines
        ax.text(lx, ly, label, ha='center', va='center', fontsize=label_size, color='#4d4d4d', 
                zorder=5, **MONO, bbox=dict(facecolor='#f9f9f7', edgecolor='none', pad=2))

# Title
ax.text(50, 97, "GAT Architecture & Training Objective", ha='center', va='center',
        fontsize=16, fontweight='bold', color='#2a2a2a')
ax.text(50, 94.5, "LM_Initialized_SkillGAT Model (Link Prediction)",
        ha='center', va='center', fontsize=11, color='#777777')

# Layout Dimensions
b_w = 60 # Box width
b_h = 8  # Box height

# --- Stage 1: Input ---
in_text = "Node Matrix (X): MiniLM Embeddings\nEdge Index & Log-normalized Edge Weights"
box(50, 86, b_w, b_h, "1. Graph Inputs", in_text, C_INPUT)

arrow(50, 82, 50, 75, label=" [2502, 384] ")

# --- Stage 2: Layer 1 ---
gat1_text = "In: 384 | Out: 32 | Heads: 4\n(4 heads × 32 = 128 output dims)"
box(50, 71, b_w, b_h, "2. GATConv Layer 1", gat1_text, C_GAT)

arrow(50, 67, 50, 60, label=" [2502, 128] ")

# --- Stage 3: Activation & Norm 1 ---
norm1_text = "1. LayerNorm(128)\n2. ELU Activation\n3. Dropout(p=0.3)"
box(50, 56, b_w, b_h, "3. Normalization & Activation", norm1_text, C_NORM)

arrow(50, 52, 50, 45, label=" [2502, 128] ")

# --- Stage 4: Layer 2 ---
gat2_text = "In: 128 | Out: 128 | Heads: 1\n(Consolidates attention into final vector)"
box(50, 41, b_w, b_h, "4. GATConv Layer 2", gat2_text, C_GAT)

arrow(50, 37, 50, 30, label=" [2502, 128] ")

# --- Stage 5: Final Norm ---
norm2_text = "1. LayerNorm(128)\n2. F.normalize(p=2, dim=-1) (L2 Norm)\n*Prevents scale collapse*"
box(50, 26, b_w, b_h, "5. Final Normalization", norm2_text, C_NORM)

arrow(50, 22, 50, 15, label=" [2502, 128] ")

# --- Stage 6: Loss Objective ---
loss_text = (
    "• Dot Product: z_u · z_v for positive/negative edges\n"
    "• BCEWithLogitsLoss: pos_weight=3.0\n"
    "• Spread Loss: -z.std(dim=0).mean() (prevents oversmoothing)"
)
box(50, 10, 75, b_h+1.5, "6. Link Prediction Loss Formulation", loss_text, C_LOSS, linespacing=1.6)

# Injecting Edge features indicator (side arrows)
ax.annotate('', xy=(32, 71), xytext=(15, 71),
            arrowprops=dict(arrowstyle='->', color=LINE, lw=1.2, linestyle='--'), zorder=2)
ax.text(15, 73, "Edge\nWeights", ha='center', va='center', fontsize=8, color='#777777')

ax.annotate('', xy=(32, 41), xytext=(15, 41),
            arrowprops=dict(arrowstyle='->', color=LINE, lw=1.2, linestyle='--'), zorder=2)
ax.text(15, 43, "Edge\nWeights", ha='center', va='center', fontsize=8, color='#777777')


# Final Touches
plt.tight_layout()
plt.savefig('architecture_skillgat.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> architecture_skillgat.png")