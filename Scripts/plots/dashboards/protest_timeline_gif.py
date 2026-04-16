"""
protest_timeline_gif.py
========================
Creates an animated GIF showing the spatial spread of zoning protest petitions
across Austin, TX from 2007 to 2024 — the "NIMBYism wildfire" visualization.

Outputs: Analysis/Output/Descriptive/nimbyism_wildfire.gif
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
try:
    # Attempt to locate the root Scripts directory
    _curr = os.path.dirname(os.path.abspath(__file__))
    while os.path.basename(_curr) != 'Scripts' and os.path.dirname(_curr) != _curr:
        _curr = os.path.dirname(_curr)
    if _curr not in sys.path:
        sys.path.insert(0, _curr)
    from thesis_style import set_thesis_style
    set_thesis_style()
except Exception:
    pass

from matplotlib.patches import Circle
import os, warnings
warnings.filterwarnings('ignore')

ROOT    = r"C:\Users\dhl\data\thesis\thesis"
OUT_DIR = os.path.join(ROOT, "Analysis", "Output", "Descriptive")
CSV     = os.path.join(OUT_DIR, "protest_timeline_geo.csv")

# ── load ──
df = pd.read_csv(CSV)
df = df.dropna(subset=['latitude', 'longitude', 'year'])
df['year'] = df['year'].astype(int)

# ── bounds (Austin metro) ──
lat_min, lat_max = df['latitude'].min() - 0.02, df['latitude'].max() + 0.02
lon_min, lon_max = df['longitude'].min() - 0.02, df['longitude'].max() + 0.02

years = sorted(df['year'].unique())
cumulative = pd.DataFrame()

frames = []
fig, ax = plt.subplots(figsize=(8, 10))

for yr in years:
    ax.clear()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    new = df[df['year'] == yr]
    cumulative = pd.concat([cumulative, new])

    # Plot historical (faded)
    old = cumulative[cumulative['year'] < yr]
    if len(old) > 0:
        # Fade by age
        old_copy = old.copy()
        old_copy['age'] = yr - old_copy['year']
        max_age = old_copy['age'].max() if old_copy['age'].max() > 0 else 1
        old_copy['alpha'] = 0.15 + 0.35 * (1 - old_copy['age'] / max_age)
        for _, row in old_copy.iterrows():
            ax.scatter(row['longitude'], row['latitude'],
                      c='#ff9999', s=15, alpha=row['alpha'], zorder=1, edgecolors='none')

    # Plot current year (bright)
    if len(new) > 0:
        ax.scatter(new['longitude'], new['latitude'],
                  c='#dc2626', s=60, alpha=0.9, edgecolors='darkred',
                  linewidth=0.8, zorder=3, label=f'{yr}: {len(new)} new')

        # "Ripple" effect — larger faint circle around new points
        ax.scatter(new['longitude'], new['latitude'],
                  c='none', s=300, alpha=0.2, edgecolors='red',
                  linewidth=1.5, zorder=2)

    # Styling
    ax.set_facecolor('#1a1a2e')
    fig.set_facecolor('#0f0f23')
    ax.set_title(f'Austin Zoning Protest Petitions — {yr}',
                fontsize=16, fontweight='bold', color='white', pad=15)

    # Stats
    total_cum = len(cumulative)
    stats_text = f'Cumulative: {total_cum} cases | New this year: {len(new)}'
    ax.text(0.5, -0.02, stats_text, transform=ax.transAxes,
           ha='center', fontsize=10, color='#aaaaaa')

    ax.set_xlabel('Longitude', color='#888888', fontsize=9)
    ax.set_ylabel('Latitude', color='#888888', fontsize=9)
    ax.tick_params(colors='#888888', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#333333')

    # Year watermark
    ax.text(0.95, 0.95, str(yr), transform=ax.transAxes,
           fontsize=48, fontweight='bold', color='white', alpha=0.15,
           ha='right', va='top')

    fig.tight_layout()
    # Save frame
    frame_path = os.path.join(OUT_DIR, f'_frame_{yr}.png')
    fig.savefig(frame_path, dpi=100, bbox_inches='tight',
               facecolor=fig.get_facecolor())
    frames.append(frame_path)
    print(f"  Frame {yr}: {len(new)} new, {total_cum} cumulative")

plt.close()

# ── Assemble GIF ──
print("\nAssembling GIF...")
try:
    from PIL import Image
    images = [Image.open(f) for f in frames]
    # Hold last frame longer
    durations = [600] * len(images)
    durations[-1] = 2000  # hold last frame 2 seconds

    gif_path = os.path.join(OUT_DIR, "nimbyism_wildfire.gif")
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                  duration=durations, loop=0, optimize=True)
    print(f"  → Saved {gif_path}")
    gif_size_mb = os.path.getsize(gif_path) / 1024 / 1024
    print(f"  → Size: {gif_size_mb:.1f} MB")
except ImportError:
    print("  ⚠ Pillow not installed. Saving frames only.")
    print("  Install with: pip install Pillow")

# Also save as static composite
print("\nSaving static composite...")
fig2, ax2 = plt.subplots(figsize=(10, 12))
ax2.set_xlim(lon_min, lon_max)
ax2.set_ylim(lat_min, lat_max)
ax2.set_facecolor('#1a1a2e')
fig2.set_facecolor('#0f0f23')

# Color by year
norm = plt.Normalize(vmin=min(years), vmax=max(years))
cmap = plt.cm.YlOrRd
sc = ax2.scatter(df['longitude'], df['latitude'],
                c=df['year'], cmap=cmap, norm=norm,
                s=25, alpha=0.8, edgecolors='none', zorder=2)
cbar = plt.colorbar(sc, ax=ax2, shrink=0.6, pad=0.02)
cbar.set_label('Year', color='white', fontsize=12)
cbar.ax.tick_params(colors='white')
ax2.set_title('All Zoning Protest Petitions in Austin (2007–2024)',
             fontsize=16, fontweight='bold', color='white', pad=15)
ax2.tick_params(colors='#888888', labelsize=8)
for spine in ax2.spines.values():
    spine.set_color('#333333')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig8_all_protests_heatmap.png"),
            dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close()
print(f"  → Saved fig8_all_protests_heatmap.png")

# Cleanup temp frames
for f in frames:
    try: os.remove(f)
    except: pass
print("\nDone!")
