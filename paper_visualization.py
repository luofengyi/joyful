import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_all_panels(save_path="paper_figure.png", dpi=400, show=False):
    """
    生成与示例图一致的4联图：
    (A) 4-way分类相关性热力图
    (B) 超参数(alpha, beta)权重3D柱状图
    (C) 窗口大小对加权F1的影响
    (D) 4-way分类混淆矩阵
    """

    # -----------------------------
    # 1) 你可以在这里替换为真实实验数据
    # -----------------------------
    labels = ["None", "FM", "EP", "GP", "FM & EP", "FM & GP"]
    corr_matrix = np.array(
        [
            [0.00, 0.23, 0.32, 0.39, 0.42, 0.38],
            [0.23, 0.22, 0.26, 0.46, 0.55, 0.56],
            [0.32, 0.26, 0.21, 0.54, 0.52, 0.53],
            [0.39, 0.46, 0.44, 0.24, 0.54, 0.52],
            [0.42, 0.55, 0.52, 0.54, 0.21, 0.66],
            [0.38, 0.56, 0.53, 0.52, 0.66, 0.56],
        ]
    )

    alpha_vals = np.array([0.2, 0.3, 0.4, 0.5])
    beta_vals = np.array([0.02, 0.06, 0.10])
    f1_grid = np.array(
        [
            [80.2, 81.6, 82.1],
            [82.8, 84.4, 85.0],
            [83.5, 85.3, 85.8],
            [82.9, 84.5, 85.1],
        ]
    )

    window_sizes = np.arange(1, 12)
    weighted_f1 = np.array([0.834, 0.845, 0.845, 0.853, 0.856, 0.840, 0.857, 0.844, 0.857, 0.857, 0.833])

    cm_labels = ["Happy", "Sad", "Neutral", "Anger"]
    conf_mat = np.array(
        [
            [110, 2, 14, 0],
            [12, 232, 22, 9],
            [21, 10, 312, 4],
            [1, 1, 36, 157],
        ]
    )

    # -----------------------------
    # 2) 统一绘图风格
    # -----------------------------
    sns.set_style("white")
    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"

    fig = plt.figure(figsize=(18, 4.8))
    gs = fig.add_gridspec(1, 4, wspace=0.45)

    # -----------------------------
    # (A) 热力图
    # -----------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=0.0,
        vmax=0.6,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.85},
        linewidths=0.4,
        linecolor="white",
        ax=ax1,
    )
    ax1.tick_params(axis="x", rotation=20)
    ax1.tick_params(axis="y", rotation=0)
    ax1.set_title("(A) IEMOCAP (4-way) Classification", fontsize=11, fontweight="bold", y=-0.28)

    # -----------------------------
    # (B) 3D 柱状图
    # -----------------------------
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    xpos, ypos = np.meshgrid(np.arange(len(alpha_vals)), np.arange(len(beta_vals)), indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)
    dx = np.full_like(xpos, 0.5, dtype=float)
    dy = np.full_like(ypos, 0.5, dtype=float)
    dz = f1_grid.ravel()

    colors = plt.cm.YlGn((dz - dz.min()) / (dz.max() - dz.min() + 1e-9))
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, edgecolor="k", linewidth=0.2)

    ax2.set_xticks(np.arange(len(alpha_vals)) + 0.25)
    ax2.set_yticks(np.arange(len(beta_vals)) + 0.25)
    ax2.set_xticklabels([f"{a:.1f}" for a in alpha_vals], fontsize=9)
    ax2.set_yticklabels([f"{b:.2f}" for b in beta_vals], fontsize=9)
    ax2.set_xlabel(r"$\alpha$", labelpad=4, fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"$\beta$", labelpad=4, fontsize=11, fontweight="bold")
    ax2.set_zlabel("Weighted F1 Score", labelpad=6, fontsize=10, fontweight="bold")
    ax2.view_init(elev=28, azim=-60)
    ax2.set_title("(B) IEMOCAP (4-way) Classification", fontsize=11, fontweight="bold", y=-0.22)

    # -----------------------------
    # (C) 直方图（柱状图）
    # -----------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(window_sizes, weighted_f1, color="#2a9d55", width=0.72, alpha=0.9)
    ax3.set_xlim(1, 11)
    ax3.set_ylim(0.81, 0.86)
    ax3.set_xticks(window_sizes)
    ax3.set_xlabel("Window Size", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Weighted F1 Score", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.set_title("(C) IEMOCAP (4-way) Window Size", fontsize=11, fontweight="bold", y=-0.28)

    # -----------------------------
    # (D) 混淆矩阵
    # -----------------------------
    ax4 = fig.add_subplot(gs[0, 3])
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="YlGn",
        square=True,
        xticklabels=cm_labels,
        yticklabels=cm_labels,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Number of Samples"},
        ax=ax4,
    )
    ax4.set_xlabel("True Label", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Predicted Label", fontsize=11, fontweight="bold")
    ax4.tick_params(axis="x", rotation=15)
    ax4.tick_params(axis="y", rotation=0)
    ax4.set_title("(D) IEMOCAP (4-way) Error Visualization", fontsize=11, fontweight="bold", y=-0.28)

    # 3D Axes 与 tight_layout 兼容性较差，使用手动边距避免警告
    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.22, wspace=0.35)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_all_panels(save_path="paper_figure.png", dpi=400, show=False)
