# -*- coding: utf-8 -*-
"""
【适用模型：Alpamayo-R1 (Macro-MoE 数据打标)】
本文件属于 Alpamayo 项目的轨迹 MoE 前期数据聚类模块。
鍩轰簬 x銆亂 鍧愭爣鐨凣T杞ㄨ抗鍧囪　鑱氱被銆?
浣跨敤涓ユ牸绛夐绾︽潫KMeans锛氭瘡绨囩簿纭垎閰?N//n_clusters 涓牱鏈€?
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import sys
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from dataset import PhysicalAIAVDatasetInterface

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def load_gt_trajectories(clip_ids, t0_offsets_us, num_future_steps=64, time_step=0.1):
    avdi = PhysicalAIAVDatasetInterface()
    dt_us = int(time_step * 1_000_000)
    trajectories, metadata = [], []
    skipped = 0

    for i, clip_id in enumerate(clip_ids):
        if (i + 1) % 100 == 0:
            logger.info(f"进度: {i+1}/{len(clip_ids)} clips, 已收集 {len(trajectories)} 条轨迹")
        try:
            egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, types="egomotion")
        except Exception as e:
            logger.warning(f"鍔犺浇clip {clip_id} 澶辫触: {e}")
            continue

        for t0_us in t0_offsets_us:
            try:
                history_ts = t0_us + np.arange(-(16 - 1) * dt_us, dt_us // 2, dt_us).astype(np.int64)
                future_ts = t0_us + np.arange(dt_us, int((num_future_steps + 0.5) * dt_us), dt_us).astype(np.int64)
                ego_hist = egomotion(history_ts)
                ego_fut = egomotion(future_ts)
                t0_xyz = ego_hist.pose.translation[-1].copy()
                t0_rot_inv = spt.Rotation.from_quat(ego_hist.pose.rotation.as_quat()[-1]).inv()
                fut_xyz_local = t0_rot_inv.apply(ego_fut.pose.translation - t0_xyz)
                if np.any(np.isnan(fut_xyz_local)) or np.any(np.isinf(fut_xyz_local)):
                    skipped += 1
                    continue
                trajectories.append(fut_xyz_local)
                metadata.append({"clip_id": clip_id, "t0_us": t0_us})
            except Exception:
                skipped += 1

    logger.info(f"共加载 {len(trajectories)} 条有效轨迹，跳过 {skipped} 条无效数据")
    return np.array(trajectories), pd.DataFrame(metadata)


def cluster_trajectories_xy(trajectories, n_clusters=3):
    """严格等频约束 KMeans：每簇精确分配 N//n_clusters 个样本。"""
    N = len(trajectories)
    quotas = [N // n_clusters + (1 if i < N % n_clusters else 0) for i in range(n_clusters)]

    xy = trajectories[:, :, :2].reshape(N, -1)
    scaler = StandardScaler()
    xy_norm = scaler.fit_transform(xy)
    X = PCA(n_components=min(32, N, xy_norm.shape[1]), random_state=42).fit_transform(xy_norm)

    # 鍒濆鍖栵細缁堢偣鏂瑰悜瑙掔瓑棰戝垎绠?
    angle = np.arctan2(trajectories[:, -1, 1], trajectories[:, -1, 0])
    sorted_idx = np.argsort(angle)
    labels = np.empty(N, dtype=int)
    pos = 0
    for k, q in enumerate(quotas):
        labels[sorted_idx[pos:pos + q]] = k
        pos += q

    # 涓ユ牸绛夐绾︽潫KMeans杩唬
    for _ in range(200):
        centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (N, K)
        new_labels = np.full(N, -1, dtype=int)
        counts = np.zeros(n_clusters, dtype=int)
        # 鎸?鍒版渶杩戜腑蹇冭窛绂?浠庡皬鍒板ぇ璐績鍒嗛厤锛屼弗鏍奸伒瀹堥厤棰?
        for idx in np.argsort(dists.min(axis=1)):
            for k in np.argsort(dists[idx]):
                if counts[k] < quotas[k]:
                    new_labels[idx] = k
                    counts[k] += 1
                    break
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    return labels, scaler, xy_norm


def visualize_cluster_trajectories(trajectories, labels, output_dir, max_per_cluster=200):
    unique_labels = sorted(np.unique(labels))
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(6 * len(unique_labels), 6), squeeze=False)
    for idx, cluster_id in enumerate(unique_labels):
        ax = axes[0][idx]
        for traj in trajectories[labels == cluster_id][:max_per_cluster]:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.15, linewidth=1)
        mean_traj = np.mean(trajectories[labels == cluster_id], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], linewidth=3, label="mean trajectory")
        ax.scatter([0], [0], marker="x", s=100, label="start")
        ax.set_title(f"Cluster {cluster_id} ({np.sum(labels == cluster_id)} trajectories)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.axis("equal"); ax.grid(True); ax.legend()
    plt.tight_layout()
    save_path = Path(output_dir) / "cluster_trajectories_xy.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"宸蹭繚瀛樿建杩瑰彔鍔犲浘: {save_path}")


def visualize_cluster_centers(trajectories, labels, output_dir):
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(np.unique(labels)):
        mean_traj = np.mean(trajectories[labels == cluster_id], axis=0)
        plt.plot(mean_traj[:, 0], mean_traj[:, 1], linewidth=3, label=f"Cluster {cluster_id}")
        plt.scatter(mean_traj[0, 0], mean_traj[0, 1], marker="x", s=80)
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("Cluster center trajectories (mean xy)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    save_path = Path(output_dir) / "cluster_centers_xy.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"宸蹭繚瀛樹腑蹇冭建杩瑰浘: {save_path}")


def visualize_pca_distribution(xy_norm, labels, output_dir):
    emb_2d = PCA(n_components=2, random_state=42).fit_transform(xy_norm)
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=10, alpha=0.5, label=f"Cluster {cluster_id}")
    plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
    plt.title("Trajectory clusters in PCA space (xy coordinates)")
    plt.grid(True); plt.legend()
    save_path = Path(output_dir) / "cluster_pca_scatter.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"宸蹭繚瀛楶CA鏁ｇ偣鍥? {save_path}")


def main():
    parser = argparse.ArgumentParser(description="鍩轰簬xy鍧愭爣鐨凣T杞ㄨ抗鍧囪　鑱氱被涓庡彲瑙嗗寲")
    parser.add_argument("--clip-index", default="clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-end", type=int, default=19)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--output-dir", default="gt_clustering_results_xy_av_3")
    parser.add_argument("--max-plot-per-cluster", type=int, default=200)
    args = parser.parse_args()

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    clip_ids = clip_index[chunk_mask].index.tolist()
    logger.info(f"浠巆hunk {args.chunk_start}-{args.chunk_end} 鍏辫幏鍙?{len(clip_ids)} 涓猚lip")

    t0_offsets_us = list(range(2_000_000, 12_000_000, 1_000_000))
    trajectories, metadata = load_gt_trajectories(clip_ids, t0_offsets_us)
    if len(trajectories) == 0:
        raise ValueError("没有加载到有效轨迹")

    logger.info(f"寮€濮嬪潎琛¤仛绫?(n_clusters={args.n_clusters}, 鐗瑰緛=xy鍧愭爣, 涓ユ牸绛夐)...")
    labels, scaler, xy_norm = cluster_trajectories_xy(trajectories, args.n_clusters)

    for i in range(args.n_clusters):
        count = np.sum(labels == i)
        logger.info(f"  绨?{i}: {count} 鏉¤建杩?({100 * count / len(labels):.1f}%)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    metadata["cluster_label"] = labels
    metadata.to_csv(output_dir / "cluster_labels.csv", index=False)
    np.save(output_dir / "trajectories.npy", trajectories)
    with open(output_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler}, f)
    logger.info(f"鑱氱被缁撴灉宸蹭繚瀛樺埌 {output_dir}/")

    visualize_cluster_trajectories(trajectories, labels, output_dir, max_per_cluster=args.max_plot_per_cluster)
    visualize_cluster_centers(trajectories, labels, output_dir)
    visualize_pca_distribution(xy_norm, labels, output_dir)
    logger.info("全部可视化完成")


if __name__ == "__main__":
    main()
