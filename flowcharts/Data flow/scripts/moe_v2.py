import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 15))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

# Palette
C_INPUT = ('#f4f3ee', '#dcd8c8', '#4d4d4d')
C_TEXT  = ('#edeafc', '#c4bdf0', '#3d2685')
C_GRAPH = ('#eafcf2', '#bdf0d4', '#1f6b45')
C_GATE  = ('#fef3fb', '#e8b8e0', '#6b1f63')
C_FINAL = ('#fcf1db', '#e8cc97', '#735012')
LINE    = '#a39f96'

def box(x, y, w, h, title, sub=None, theme=C_INPUT):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.0,rounding_size=1.5', 
                                facecolor=theme[0], edgecolor=theme[1], linewidth=1.5, zorder=3))
    ax.text(x, y + (h*0.12 if sub else 0), title, ha='center', va='center', fontsize=12, fontweight='bold', color=theme[2], zorder=4)
    if sub:
        ax.text(x, y - h*0.22, sub, ha='center', va='center', fontsize=9.5, color=theme[2], alpha=0.8, zorder=4)

def ortho_arrow(points, color=LINE):
    """Draws a strictly orthogonal path with an arrowhead."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=1.6, zorder=2)
    # Arrowhead at the last segment
    ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]), 
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.6, mutation_scale=15), zorder=3)

# --- Title ---
ax.text(50, 97, "Conceptual Data Flow — Mixture of Experts v2", ha='center', va='center', fontsize=16, fontweight='bold', color='#2a2a2a')
ax.text(50, 94.5, "High-level logical stages of the Resume-JD matching pipeline", ha='center', va='center', fontsize=10, color='#777777')

# --- 1. Input Stage ---
box(50, 88, 60, 7, "1. Multimodal Input", "Raw Resume & Job Description Data", C_INPUT)

# Branching logic
ortho_arrow([(50, 84.5), (50, 82), (25, 82), (25, 78)]) # To Text
ortho_arrow([(50, 84.5), (50, 82), (75, 82), (75, 78)]) # To Graph

# --- 2. Feature Extraction ---
box(25, 72, 40, 12, "2A. Semantic Analysis", "Extracting deep contextual meaning\nfrom natural language sentences", C_TEXT)
box(75, 72, 40, 12, "2B. Skill Graph Analysis", "Mapping specific skill entities to a\npretrained global knowledge graph", C_GRAPH)

ortho_arrow([(25, 66), (25, 61)])
ortho_arrow([(75, 66), (75, 61)])

# --- 3. Expert Matching ---
box(25, 54, 40, 14, "3A. Holistic Matcher", "Scores the 'vibe' and overall\nexperience alignment", C_TEXT)
box(75, 54, 40, 14, "3B. Granular Skill Matcher", "Scores exact technical coverage\nusing context-aware attention", C_GRAPH)

# --- Confidence / Skill Count Path ---
# High level: This represents the 'Certainty' signal
box(50, 42, 22, 6, "Confidence Signal", "Resume Skill Density", C_INPUT)
# Orthogonal path from input to confidence
ortho_arrow([(80, 88), (95, 88), (95, 42), (61, 42)])

# --- 4. Gating Logic ---
# Arrows from scores and signal to the gate
ortho_arrow([(25, 47), (25, 36), (40, 36), (40, 33)]) # Text Score -> Gate
ortho_arrow([(75, 47), (75, 36), (60, 36), (60, 33)]) # Graph Score -> Gate
ortho_arrow([(50, 39), (50, 33)])                      # Signal -> Gate

box(50, 27, 45, 12, "4. Adaptive Gating Logic", "Decides which Expert to trust more\nbased on the richness of resume skills", C_GATE)

# --- 5. Integrated Output ---
# Route mixing weights to output
ortho_arrow([(50, 21), (50, 14)])
# Route scores directly to final sum
ortho_arrow([(25, 47), (25, 14), (32, 14)])
ortho_arrow([(75, 47), (75, 14), (68, 14)])

box(50, 14, 80, 6, "5. Unified Match Prediction", "A weighted synthesis of Semantic and Technical evidence", C_FINAL)

# --- 6. Optimization ---
ortho_arrow([(50, 11), (50, 7)])
box(50, 4, 60, 6, "6. Multi-Objective Learning", "Continuous improvement via ranking & accuracy loss", C_INPUT)

plt.tight_layout()
plt.savefig('data_flow_moe_v2.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> data_flow_moe_v2.png")