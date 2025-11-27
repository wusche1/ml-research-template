import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_all():
    ratings_dir = Path("data/ratings")
    monitors = [d.name for d in ratings_dir.iterdir() if d.is_dir()]
    
    fig, axes = plt.subplots(len(monitors), 3, figsize=(16, 4 * len(monitors)), squeeze=False,
                             gridspec_kw={'width_ratios': [1, 1, 1.2]})
    
    handles, labels = [], []
    for row, monitor in enumerate(monitors):
        monitor_dir = ratings_dir / monitor
        datasets = {d.name: json.load(open(d / "ratings.json")) for d in monitor_dir.iterdir() if d.is_dir()}
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
        
        # Load config if exists
        config_path = monitor_dir / "config.json"
        config = json.load(open(config_path)) if config_path.exists() else {"model": "unknown", "prompt": ""}
        
        # Boxplot with colors
        ax_box = axes[row, 0]
        data = [[r["rating"] for r in ratings] for ratings in datasets.values()]
        bp = ax_box.boxplot(data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_box.set_xticks([])
        ax_box.set_ylabel("Rating")
        ax_box.set_title(f"{monitor} - Boxplot")
        
        # Point cloud
        ax_scatter = axes[row, 1]
        for i, (name, ratings) in enumerate(datasets.items()):
            scores = [r["rating"] for r in ratings]
            x = np.random.normal(i, 0.1, len(scores))
            h = ax_scatter.scatter(x, scores, alpha=0.5, color=colors[i], label=name, s=20)
            if row == 0:
                handles.append(h)
                labels.append(name)
        ax_scatter.set_xticks([])
        ax_scatter.set_ylabel("Rating")
        ax_scatter.set_title(f"{monitor} - Point Cloud")
        
        # Config text boxes
        ax_text = axes[row, 2]
        ax_text.axis('off')
        
        # Model box
        ax_text.text(0.05, 0.98, "Model:", transform=ax_text.transAxes, fontsize=9, fontweight='bold', verticalalignment='top')
        ax_text.text(0.05, 0.92, config['model'], transform=ax_text.transAxes, fontsize=8, fontfamily='monospace',
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # Prompt box
        ax_text.text(0.05, 0.82, "Prompt:", transform=ax_text.transAxes, fontsize=9, fontweight='bold', verticalalignment='top')
        prompt_preview = config["prompt"][:400] + "..." if len(config["prompt"]) > 400 else config["prompt"]
        ax_text.text(0.05, 0.76, prompt_preview, transform=ax_text.transAxes, fontsize=7, fontfamily='monospace',
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    Path("plots").mkdir(exist_ok=True)
    plt.savefig("plots/ratings.png", dpi=150, bbox_inches='tight')
    print("Saved plots/ratings.png")

if __name__ == "__main__":
    plot_all()
