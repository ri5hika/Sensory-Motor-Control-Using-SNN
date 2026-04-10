import matplotlib.pyplot as plt
import numpy as np

# Use a modern style
plt.style.use("seaborn-v0_8-whitegrid")

# Domains
domains = ["Scheduling", "Email Automation", "Healthcare", "Therapy"]
x = np.arange(len(domains))

# Metrics data
metrics = {
    "Task Completion Rate (%)": ([92.5, 94.0, 85.0, 78.5], 100),
    "Task Completion Time (s)": ([12.6, 14.1, 32.9, 41.5], 50),
    "Step Accuracy (%)": ([95.0, 93.0, 88.0, 82.0], 100),
    "Resource Utilization (%)": ([35, 40, 65, 75], 100),
    "Error / Hallucination Rate (%)": ([4.0, 5.0, 11.0, 15.0], 20),
    "Recovery Rate (%)": ([90.0, 88.0, 80.0, 72.0], 100),
}

# Plot settings
bar_width = 0.6
corner_radius = 6

for title, (values, y_max) in metrics.items():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        x,
        values,
        width=bar_width,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    # Titles and labels
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_ylabel(title, fontsize=11)
    ax.set_xlabel("Domain", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=10)
    ax.set_ylim(0, y_max)

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_max * 0.02,
            f"{height}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold"
        )

    # Subtle grid
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", visible=False)

    plt.tight_layout()
    plt.show()
