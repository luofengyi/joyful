from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def split_four_panels(
    input_path="paper_figure.png",
    out_prefix="paper_panel",
    left_trim_ratio=0.0,
    right_trim_ratio=0.0,
):
    """
    将四联图按横向平均切分成四个独立图片。
    可通过 left/right_trim_ratio 先裁掉左右留白（0~0.2 常用）。
    """
    img_path = Path(input_path)
    if not img_path.exists():
        raise FileNotFoundError(f"输入图片不存在: {input_path}")

    img = mpimg.imread(img_path)
    h, w = img.shape[:2]

    left = int(w * left_trim_ratio)
    right = int(w * (1 - right_trim_ratio))
    work = img[:, left:right]
    ww = work.shape[1]

    # 等宽切成4份
    bounds = [0, ww // 4, ww // 2, ww * 3 // 4, ww]
    names = ["A", "B", "C", "D"]

    outputs = []
    for i in range(4):
        crop = work[:, bounds[i] : bounds[i + 1]]
        out_file = f"{out_prefix}_{names[i]}.png"
        plt.imsave(out_file, crop)
        outputs.append(out_file)

    return outputs


if __name__ == "__main__":
    files = split_four_panels(
        input_path="paper_figure.png",
        out_prefix="paper_panel",
        left_trim_ratio=0.0,
        right_trim_ratio=0.0,
    )
    for f in files:
        print(f"Saved: {f}")
