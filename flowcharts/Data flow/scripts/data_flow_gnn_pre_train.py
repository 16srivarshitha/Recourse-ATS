import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Canvas Setup
fig, ax = plt.subplots(figsize=(12, 14)) # Slightly taller to accommodate the deep GNN pipeline
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

# Custom Color Palettes
C_INPUT   = ('#eaf2f8', '#b5d0e8', '#28547a') # Soft Blue (JSON Inputs)
C_MINILM  = ('#edeafc', '#c4bdf0', '#3d2685') # Purple (Language Model)
C_GRAPH   = ('#fef3fb', '#e8b8e0', '#6b1f63') # Pink (Graph Sparsification)
C_PYG     = ('#e6f5ea', '#a9d1b5', '#1e522d') # Soft Green (PyTorch Geometric)
C_MODEL   = ('#f4f3ee', '#dcd8c8', '#4d4d4d') # Grey (Model Architecture)
C_TRAIN   = ('#fcedec', '#ebb0ad', '#7a221f') # Coral/Red (Training & Loss)
C_FINAL   = ('#fcf1db', '#e8cc97', '#735012') # Gold (Outputs)
LINE      = '#a39f96'
MONO      = {'fontfamily': 'monospace'}

# Helpers
def box(x, y, w, h, title, sub=None, theme=C_INPUT, title_size=10.5, sub_size=8.5, linespacing=1.3):
    fc, ec, tc = theme
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.0,rounding_size=1.5',
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    if sub:
        ax.text(x, y + h * 0.15, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)
        ax.text(x, y - h * 0.20, sub, ha='center', va='center',
                fontsize=sub_size, color=tc, alpha=0.85, zorder=4, linespacing=linespacing)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=tc, zorder=4)

def arrow(x1, y1, x2, y2, ls='-', lw=1.4, color=LINE, label=None, label_size=8.5, label_pos=0.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle=ls, mutation_scale=13), zorder=2)
    if label:
        lx = x1 + (x2 - x1) * label_pos
        ly = y1 + (y2 - y1) * label_pos
        ax.text(lx, ly, label, ha='center', va='center', fontsize=label_size, color=color, 
                zorder=5, **MONO, bbox=dict(facecolor='#f9f9f7', edgecolor='none', pad=1))

# Title
ax.text(50, 97, "Data Preparation & Extraction Flow", ha='center', va='center',
        fontsize=15, fontweight='bold', color='#2a2a2a')
ax.text(50, 94.5, "Notebook 2: GNN Pre-training via Link Prediction",
        ha='center', va='center', fontsize=10, color='#777777')

# Layout Dimensions
b_w_half = 40 # Box width for twin columns
b_w_full = 65 # Box width for center columns
b_h = 8.5     # Box height

# --- ROW 1: Raw Inputs (From NB 1) ---
box(25, 87, b_w_half, b_h, "1A. Node Vocabulary", "skill_vocab.json\n(2,502 Unique Skills)", C_INPUT)
box(75, 87, b_w_half, b_h, "1B. Co-occurrence Edges", "graph_edges.json\n(740,444 Raw Pairs)", C_INPUT)

arrow(25, 82.5, 25, 76.5)
arrow(75, 82.5, 75, 76.5)

# --- ROW 2: Data Pre-processing ---
box(25, 72, b_w_half, b_h, "2A. Inductive Initialization", "MiniLM Text Embeddings [CLS]\nProduces 384-dim Node Matrix (X)", C_MINILM)
box(75, 72, b_w_half, b_h, "2B. Graph Sparsification", "Min co-occurrence >= 10\nLog-normalized edge weights", C_GRAPH)

# Merging into PyG Data
arrow(25, 67.5, 40, 61.5, label=" X Matrix", label_pos=0.3)
arrow(75, 67.5, 60, 61.5, label=" Edges ", label_pos=0.3)

# --- ROW 3: PyTorch Geometric ---
box(50, 57, b_w_full, b_h, "3. PyTorch Geometric Integration", "Data(x, edge_index, edge_attr)\n163,746 Directed Edges on 2,502 Nodes", C_PYG)

arrow(50, 52.5, 50, 46.5)

# --- ROW 4: GNN Architecture ---
gnn_text = "Layer 1: GATConv (384 → 128, 4 Heads) + LayerNorm + ELU\nLayer 2: GATConv (128 → 128, 1 Head) + LayerNorm + L2 Norm"
box(50, 42, b_w_full, b_h, "4. LM_Initialized_SkillGAT", gnn_text, C_MODEL, linespacing=1.6)

arrow(50, 37.5, 50, 31.5)

# --- ROW 5: Training ---
train_text = "Weighted BCE (pos_weight=3.0) + 2x Negative Sampling\nSpread Loss to prevent embedding collapse"
box(50, 27, b_w_full, b_h, "5. Link Prediction Pre-training", train_text, C_TRAIN, linespacing=1.6)

arrow(50, 22.5, 50, 16.5, label=" Validation AUC: ~0.89 ", label_pos=0.45)

# --- ROW 6: Final Outputs ---
final_text = (
    "1. skill_embeddings.npy: Contextualized 128-dim vectors\n"
    "2. pretrained_skill_gat.pth: Model Weights for inference"
)
box(50, 11, 75, b_h + 1, "6. Final Graph Artifacts Generated", final_text, C_FINAL, linespacing=1.8)

# Final Touches
plt.tight_layout()
plt.savefig('data_flow_notebook2.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> data_flow_notebook2.png")
