import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

#  Canvas 
fig, ax = plt.subplots(figsize=(10, 12)) # Adjusted aspect ratio for the vertical flow
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

#  Palette (reused from your design) 
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT = ('#edeafc', '#c4bdf0', '#3d2685')
C_MLP = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE = '#a39f96'
MONO = {'fontfamily': 'monospace'}

#  Helpers (reused with adjustments) 
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
ax.text(50, 96, "Conceptual Data Flow for Pure Semantic", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#2a2a2a')
ax.text(50, 92.5, "Detailed data transformations and model stages",
        ha='center', va='center', fontsize=9, color='#777777')

#  New Flow Diagram 
# Define common dimensions
b_w = 45
b_h = 7

# Step 1: Input Data
box(50, 85, b_w, b_h, "1. Input Data", "Resume + Job Description (JD) text", C_INPUT)

# Transition 1
arrow(50, 81.5, 50, 75.5, label="concat as single 512-token sequence")

# Step 2: Formatted Sequence
box(50, 72, b_w, b_h, "2. Formatted Input", "[Resume text] [SEP] [JD text]", C_INPUT)

# Transition 2
arrow(50, 68.5, 50, 62.5, label="Input into model")

# Step 3: Cross-Encoder Processing
box(50, 59, b_w, b_h, "3. Cross-Encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2", C_TEXT)

# Transition 3
arrow(50, 55.5, 50, 49.5, label="[CLS] token embedding (384-dim)")

# Step 4: Embedding Extraction
box(50, 46, b_w, b_h, "4. Embedding Vector", "[CLS] Token Embedding (384 float values)", C_TEXT)

# Transition 4
arrow(50, 42.5, 50, 36.5, label="Feed to prediction head")

# Step 5: MLP Head
box(50, 31, b_w, b_h, "5. Prediction Head (MLP)",
    "Linear(384->64) -> ReLU -> Linear(64->1) -> Sigmoid", C_MLP, linespacing=1.1)

# Transition 5
arrow(50, 27.5, 50, 21.5, label="scalar")

# Step 6: Final Output
box(50, 18, b_w, b_h, "6. Match Score", "Scalar value ∈ [0, 1]", C_FINAL)

#  Final Touches 
plt.tight_layout()
# Note: In a script environment, you would use plt.show() or plt.savefig()
# plt.show()
print("Conceptual data flow chart script created.")
plt.tight_layout()
plt.savefig('data_flow_pure_semantic.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> data_flow_pure_semantic.png")
