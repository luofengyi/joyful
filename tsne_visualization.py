import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def pairwise_class_distance(emb2d: np.ndarray, labels: np.ndarray) -> float:
    """计算各类别中心两两欧式距离的平均值。"""
    classes = np.unique(labels)
    centers = []
    for cls in classes:
        cls_points = emb2d[labels == cls]
        if len(cls_points) == 0:
            continue
        centers.append(np.mean(cls_points, axis=0))

    if len(centers) < 2:
        return 0.0

    centers = np.array(centers)
    dists = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dists.append(np.linalg.norm(centers[i] - centers[j]))
    return float(np.mean(dists))


def build_palette(n_classes: int):
    """构建与论文图接近的高对比配色。"""
    base = [
        "#ef4444",  # red
        "#f59e0b",  # orange
        "#eab308",  # yellow
        "#10b981",  # emerald
        "#06b6d4",  # cyan
        "#3b82f6",  # blue
        "#8b5cf6",  # violet
        "#ec4899",  # pink
    ]
    if n_classes <= len(base):
        return base[:n_classes]
    cmap = plt.cm.get_cmap("tab20", n_classes)
    return [cmap(i) for i in range(n_classes)]


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names=None,
    title: str = "JOYFUL",
    save_path: str = "tsne_joyful.png",
    random_state: int = 42,
):
    """
    绘制 t-SNE 散点图并显示类别中心平均距离。
    """
    if features.ndim != 2:
        raise ValueError("features 必须是二维数组: [样本数, 特征维度]")
    if labels.ndim != 1:
        raise ValueError("labels 必须是一维数组: [样本数]")
    if len(features) != len(labels):
        raise ValueError("features 与 labels 样本数不一致")

    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(features) // 10)),
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb2d = tsne.fit_transform(features_std)

    # 归一化到更美观的显示范围，便于论文排版
    emb2d = (emb2d - emb2d.mean(axis=0)) / (emb2d.std(axis=0) + 1e-8)
    emb2d = emb2d * 15

    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [str(x) for x in unique_labels]
    palette = build_palette(len(unique_labels))

    plt.rcParams["font.family"] = "DejaVu Serif"
    fig, ax = plt.subplots(figsize=(8.2, 6.2), facecolor="#efeff4")
    ax.set_facecolor("#efeff4")

    for idx, cls in enumerate(unique_labels):
        pts = emb2d[labels == cls]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=22,
            c=[palette[idx]],
            alpha=0.9,
            edgecolors="white",
            linewidths=0.3,
            label=class_names[idx],
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.02,
        0.96,
        title,
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        va="top",
        ha="left",
    )

    avg_dist = pairwise_class_distance(emb2d, labels)
    ax.text(
        0.5,
        -0.08,
        f"Average Distance: {avg_dist:.2f}",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="center",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return save_path, avg_dist


def load_data(features_path: str, labels_path: str):
    """支持 .npy 或 .csv 的特征/标签加载。"""
    features_file = Path(features_path)
    labels_file = Path(labels_path)

    if features_file.suffix.lower() == ".npy":
        features = np.load(features_file)
    else:
        features = np.loadtxt(features_file, delimiter=",")

    if labels_file.suffix.lower() == ".npy":
        labels = np.load(labels_file)
    else:
        labels = np.loadtxt(labels_file, delimiter=",")

    labels = labels.astype(int)
    return features, labels


def demo_data(n_classes=6, points_per_class=80, dim=64, random_state=42):
    """构造演示数据，便于你无缝跑通脚本。"""
    rng = np.random.default_rng(random_state)
    centers = rng.normal(0, 4.0, size=(n_classes, dim))
    x_list, y_list = [], []
    for c in range(n_classes):
        pts = centers[c] + rng.normal(0, 1.2, size=(points_per_class, dim))
        x_list.append(pts)
        y_list.append(np.full(points_per_class, c))
    return np.vstack(x_list), np.concatenate(y_list)


def parse_args():
    parser = argparse.ArgumentParser(description="论文 t-SNE 可视化脚本")
    parser.add_argument("--features", type=str, default="", help="特征文件路径 (.npy 或 .csv)")
    parser.add_argument("--labels", type=str, default="", help="标签文件路径 (.npy 或 .csv)")
    parser.add_argument("--title", type=str, default="JOYFUL")
    parser.add_argument("--output", type=str, default="tsne_joyful.png")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.features and args.labels:
        feats, labs = load_data(args.features, args.labels)
    else:
        feats, labs = demo_data()

    out_file, avg = plot_tsne(
        features=feats,
        labels=labs,
        title=args.title,
        save_path=args.output,
        random_state=args.seed,
    )
    print(f"Saved: {out_file}")
    print(f"Average Distance: {avg:.2f}")
