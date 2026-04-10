import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Canvas Setup
fig, ax = plt.subplots(figsize=(12, 13)) # Taller aspect ratio for 6 stages
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#f9f9f7')
ax.set_facecolor('#f9f9f7')
plt.rcParams['font.family'] = 'sans-serif'

# Custom Color Palettes
C_PRIMARY  = ('#f4f3ee', '#dcd8c8', '#4d4d4d') # Warm Grey (Primary Data)
C_LINKEDIN = ('#eaf2f8', '#b5d0e8', '#28547a') # Soft Blue (LinkedIn Data)
C_PROCESS  = ('#edeafc', '#c4bdf0', '#3d2685') # Purple (Dual Pipeline)
C_EXTRACT  = ('#fef3fb', '#e8b8e0', '#6b1f63') # Pink (Extraction)
C_FINAL    = ('#fcf1db', '#e8cc97', '#735012') # Gold (Outputs)
LINE       = '#a39f96'
MONO       = {'fontfamily': 'monospace'}

# Helpers
def box(x, y, w, h, title, sub=None, theme=C_PRIMARY, title_size=10.5, sub_size=8.5, linespacing=1.3):
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
ax.text(50, 94.5, "Notebook 1: Filtering, Smart Parsing, and Vocabulary Generation",
        ha='center', va='center', fontsize=10, color='#777777')

# Layout Dimensions
b_w = 38 # Box width for twin columns
b_h = 8  # Box height

# --- ROW 1: Raw Inputs ---
box(25, 88, b_w, b_h, "1A. Primary Dataset", "Resume + Job Description (JD) Text\n(N = 6.2k)", C_PRIMARY)
box(75, 88, b_w, b_h, "1B. LinkedIn Dataset", "1.3M Real-world Job Postings", C_LINKEDIN)

arrow(25, 84, 25, 79)
arrow(75, 84, 75, 79)

# --- ROW 2: Filtering ---
box(25, 75, b_w, b_h, "2A. Domain Filter", "Keep 6.1k IT/Software JDs", C_PRIMARY)
box(75, 75, b_w, b_h, "2B. Domain Filter", "Keep 40k IT/Software Jobs", C_LINKEDIN)

arrow(25, 71, 25, 66)
arrow(75, 71, 75, 66)

# --- ROW 3: Parsing / Vocab ---
box(25, 62, b_w, b_h, "3A. Smart JD Parsing", "Bring 'Requirements' section to front\n(Bypasses 512-token limit)", C_PRIMARY)
box(75, 62, b_w, b_h, "3B. Vocab Construction", "Top 2,500 Tech Skills + Synonyms\n(Removes Soft Skills)", C_LINKEDIN)

arrow(25, 58, 25, 53)
arrow(75, 58, 75, 53)

# --- ROW 4: Dual Pipeline & Edges ---
box(25, 49, b_w, b_h, "4A. Dual Text Pipeline", "Raw (Transformer) & Cleaned (GNN)", C_PROCESS)
box(75, 49, b_w, b_h, "4B. Graph Edge Gen", "Compute 740k Skill Co-occurrences", C_LINKEDIN)

# --- CROSS-ROUTING TO EXTRACTION ---
# Arrow from Vocab (3B) to Extraction (5) - Bypasses 4B using the inner channel
arrow(56, 62, 51, 39, label=" 2.5k Vocab ", label_pos=0.45)
# Arrow from Dual Pipeline (4A) to Extraction (5)
arrow(44, 49, 49, 39, label=" Clean Text ", label_pos=0.45)

# --- ROW 5: Extraction ---
box(50, 35, 48, b_h, "5. FlashText Extraction", "O(N) mapping of 2.5k skills\nonto Resumes & JDs", C_EXTRACT)

# --- ROUTING TO OUTPUTS ---
# Straight down paths. The side channels (X=25, X=75) safely bypass Box 5 (which spans X=26 to 74).
arrow(25, 45, 25, 21, label=" Raw Text \n(CSV)", label_pos=0.7)
arrow(75, 45, 75, 21, label=" JSON \nEdges ", label_pos=0.7)
arrow(50, 31, 50, 21, label=" Extracted \nSkills ", label_pos=0.5)

# --- ROW 6: Final Outputs ---
final_text = (
    "1. train_clean.csv / test_clean.csv (Transformer & Node Features)\n"
    "2. skill_vocab.json (2,502 Entities)\n"
    "3. graph_edges.json (740,444 Connections)"
)
box(50, 14, 82, 14, "6. Final Artifacts Generated", final_text, C_FINAL, linespacing=1.8)

# Final Touches
plt.tight_layout()
plt.savefig('data_flow_notebook1.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved -> data_flow_notebook1.png")