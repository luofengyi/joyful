import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

import joyful


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def resolve_data_path(stored_args):
    if getattr(stored_args, "emotion", None):
        return os.path.join(
            stored_args.data_dir_path,
            stored_args.dataset,
            f"data_{stored_args.dataset}_{stored_args.emotion}.pkl",
        )
    if getattr(stored_args, "transformers", False):
        return os.path.join(
            stored_args.data_dir_path,
            stored_args.dataset,
            "transformers",
            f"data_{stored_args.dataset}.pkl",
        )
    return os.path.join(
        stored_args.data_dir_path,
        stored_args.dataset,
        f"data_{stored_args.dataset}.pkl",
    )


def main(args):
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    stored_args = ckpt["args"]
    stored_args.device = args.device
    stored_args.batch_size = args.batch_size

    model = ckpt["modelN_state_dict"].to(args.device)
    model_f = ckpt["modelF_state_dict"].to(args.device)
    model.eval()
    model_f.eval()

    data_path = resolve_data_path(stored_args)
    data = load_pkl(data_path)
    if args.split not in data:
        raise ValueError(f"split='{args.split}' 不存在，可选: {list(data.keys())}")

    dataset = joyful.Dataset(data[args.split], model_f, False, stored_args)

    features_all = []
    labels_all = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"export {args.split}"):
            batch = dataset[idx]
            labels = batch["label_tensor"].cpu().numpy()

            for k, v in batch.items():
                if k != "utterance_texts":
                    batch[k] = v.to(args.device)

            graph_out, seq_features, _ = model.get_rep(batch, False)

            if args.feature_source == "graph":
                features = graph_out.detach().cpu().numpy()
            elif args.feature_source == "seq":
                features = seq_features.detach().cpu().numpy()
            else:
                features = torch.cat([seq_features, graph_out], dim=-1).detach().cpu().numpy()

            if features.shape[0] != labels.shape[0]:
                raise RuntimeError(
                    f"特征数量({features.shape[0]})与标签数量({labels.shape[0]})不一致，"
                    "请检查模型输出与标签对齐方式。"
                )

            features_all.append(features)
            labels_all.append(labels)

    os.makedirs(args.output_dir, exist_ok=True)
    features_np = np.concatenate(features_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)

    feat_path = os.path.join(args.output_dir, "features.npy")
    label_path = os.path.join(args.output_dir, "labels.npy")
    np.save(feat_path, features_np)
    np.save(label_path, labels_np)

    print(f"Saved: {feat_path} shape={features_np.shape}")
    print(f"Saved: {label_path} shape={labels_np.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export features/labels for t-SNE.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path, e.g. model_checkpoints/iemocap_4_best_dev_f1_model_atv.pt",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--feature_source", type=str, default="graph", choices=["graph", "seq", "concat"])
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
