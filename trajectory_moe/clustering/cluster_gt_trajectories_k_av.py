# -*- coding: utf-8 -*-
"""
【适用模型：Alpamayo-R1 (Macro-MoE 数据打标)】
本文件属于 Alpamayo 项目的轨迹 MoE 前期数据聚类模块。
璇诲彇鍓?0涓猚hunk鐨勮建杩规暟鎹紝瀵笹T杞ㄨ抗杩涜鑱氱被骞跺彲瑙嗗寲銆?

鍩轰簬 train_decoder_expert.py 鐨勬暟鎹鍙栨柟寮忥細
- 鍓?0涓猚hunk鐨勬墍鏈塩lip
- 浠庣2绉掑埌绗?0绉掗噰鏍?
- 鎻愬彇鏈潵6.4绉?64姝ワ紝姣?.1s)鐨凣T杞ㄨ抗
- 浣跨敤瀹樻柟 UnicycleAccelCurvatureActionSpace 鎻愬彇鏇茬巼
- 鍙熀浜庢洸鐜囪繘琛孠Means鑱氱被
- 鍙鍖栦粛浣跨敤xy杞ㄨ抗鍧愭爣
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
import torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

from dataset import PhysicalAIAVDatasetInterface
from alpamayo_r1.action_space import UnicycleAccelCurvatureActionSpace

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def build_action_space():
    """使用与 config.json 一致的参数构建 action_space。"""
    return UnicycleAccelCurvatureActionSpace(
        accel_mean=0.02902694707164455,
        accel_std=0.6810426736454882,
        curvature_mean=0.0002692167976330542,
        curvature_std=0.026148280660833106,
        accel_bounds=(-9.8, 9.8),
        curvature_bounds=(-0.33, 0.33),
        dt=0.1,
        n_waypoints=64,
        theta_lambda=1e-6,
        theta_ridge=1e-8,
        v_lambda=1e-6,
        v_ridge=1e-4,
        a_lambda=1e-4,
        a_ridge=1e-4,
        kappa_lambda=1e-4,
        kappa_ridge=1e-4,
    )

def load_gt_trajectories(clip_ids, t0_offsets_us, num_future_steps=64, time_step=0.1):
    """加载 GT 轨迹，同时返回 history/future 的 xyz 和 rot。"""
    avdi = PhysicalAIAVDatasetInterface()
    dt_us = int(time_step * 1_000_000)
    trajectories, hist_xyz_list, hist_rot_list, fut_rot_list, metadata = [], [], [], [], []
    skipped = 0

    for i, clip_id in enumerate(clip_ids):
        if (i + 1) % 100 == 0:
            logger.info(f"进度: {i+1}/{len(clip_ids)} clips, 已收集 {len(trajectories)} 条轨迹")
        try:
            egomotion = avdi.get_clip_feature(
                clip_id, avdi.features.LABELS.EGOMOTION, types="egomotion"
            )
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
                t0_rot = spt.Rotation.from_quat(ego_hist.pose.rotation.as_quat()[-1])
                t0_rot_inv = t0_rot.inv()

                hist_xyz_local = t0_rot_inv.apply(ego_hist.pose.translation - t0_xyz)
                fut_xyz_local = t0_rot_inv.apply(ego_fut.pose.translation - t0_xyz)
                hist_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_hist.pose.rotation.as_quat())).as_matrix()
                fut_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_fut.pose.rotation.as_quat())).as_matrix()

                if np.any(np.isnan(fut_xyz_local)) or np.any(np.isinf(fut_xyz_local)):
                    skipped += 1
                    continue

                trajectories.append(fut_xyz_local)
                hist_xyz_list.append(hist_xyz_local)
                hist_rot_list.append(hist_rot_local)
                fut_rot_list.append(fut_rot_local)
                metadata.append({"clip_id": clip_id, "t0_us": t0_us})
            except Exception:
                skipped += 1

    logger.info(f"共加载 {len(trajectories)} 条有效轨迹，跳过 {skipped} 条无效数据")
    return (
        np.array(trajectories), np.array(hist_xyz_list),
        np.array(hist_rot_list), np.array(fut_rot_list),
        pd.DataFrame(metadata),
    )

def compute_actions(action_space, trajectories, hist_xyz, hist_rot, fut_rot, batch_size=512):
    """使用官方 action_space 批量计算（加速度、曲率）。"""
    N = len(trajectories)
    all_actions = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        with torch.no_grad():
            actions = action_space.traj_to_action(
                traj_history_xyz=torch.from_numpy(hist_xyz[start:end]).float(),
                traj_history_rot=torch.from_numpy(hist_rot[start:end]).float(),
                traj_future_xyz=torch.from_numpy(trajectories[start:end]).float(),
                traj_future_rot=torch.from_numpy(fut_rot[start:end]).float(),
            )
        all_actions.append(actions.numpy())
    return np.concatenate(all_actions, axis=0)  # (N, 64, 2)

def cluster_trajectories(actions, n_clusters=3):
    """基于曲率均值的分位数分箱，获得更平衡的聚类。"""
    curvature = actions[:, :, 1]  # (N, 64) 鍙彇鏇茬巼鍒?
    
    # 浣跨敤鏇茬巼鍧囧€间綔涓鸿仛绫荤壒寰侊紙姝?宸﹁浆锛岃礋=鍙宠浆锛?=鐩磋锛?
    mean_curvature = np.mean(curvature, axis=1)  # (N,)
    
    # 鍩轰簬鍒嗕綅鏁板垎绠憋紝纭繚姣忎釜绨囧ぇ灏忕浉杩?
    percentiles = np.linspace(0, 100, n_clusters + 1)
    bins = np.percentile(mean_curvature, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    labels = np.digitize(mean_curvature, bins[1:-1])  # 0 鍒?n_clusters-1
    
    # 涓轰簡鍏煎鍚庣画鍙鍖栵紝浠嶇劧杩斿洖鏍囧噯鍖栫殑鏇茬巼搴忓垪
    scaler = StandardScaler()
    flat_norm = scaler.fit_transform(curvature)
    
    return labels, None, scaler, flat_norm

def visualize_cluster_trajectories(trajectories, labels, output_dir, max_per_cluster=200):
    unique_labels = sorted(np.unique(labels))
    n_clusters = len(unique_labels)
    fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 6), squeeze=False)
    axes = axes[0]
    for idx, cluster_id in enumerate(unique_labels):
        ax = axes[idx]
        cluster_trajs = trajectories[labels == cluster_id]
        if len(cluster_trajs) > max_per_cluster:
            cluster_trajs = cluster_trajs[:max_per_cluster]
        for traj in cluster_trajs:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.15, linewidth=1)
        mean_traj = np.mean(trajectories[labels == cluster_id], axis=0)
        ax.plot(mean_traj[:, 0], mean_traj[:, 1], linewidth=3, label="mean trajectory")
        ax.scatter([0], [0], marker="x", s=100, label="start")
        ax.set_title(f"Cluster {cluster_id} ({np.sum(labels == cluster_id)} trajectories)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    save_path = Path(output_dir) / "cluster_trajectories_xy.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"宸蹭繚瀛樿建杩瑰彔鍔犲浘: {save_path}")

def visualize_cluster_centers(trajectories, labels, output_dir):
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(np.unique(labels)):
        mean_traj = np.mean(trajectories[labels == cluster_id], axis=0)
        plt.plot(mean_traj[:, 0], mean_traj[:, 1], linewidth=3, label=f"Cluster {cluster_id}")
        plt.scatter(mean_traj[0, 0], mean_traj[0, 1], marker="x", s=80)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Cluster center trajectories (mean xy)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    save_path = Path(output_dir) / "cluster_centers_xy.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"宸蹭繚瀛樹腑蹇冭建杩瑰浘: {save_path}")

def visualize_pca_distribution(flat_norm, labels, output_dir):
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(flat_norm)
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=10, alpha=0.5, label=f"Cluster {cluster_id}")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Trajectory clusters in PCA space (curvature)")
    plt.grid(True)
    plt.legend()
    save_path = Path(output_dir) / "cluster_pca_scatter.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"宸蹭繚瀛楶CA鏁ｇ偣鍥? {save_path}")

def visualize_z_profile(trajectories, labels, output_dir):
    plt.figure(figsize=(8, 6))
    t = np.arange(trajectories.shape[1]) * 0.1
    for cluster_id in sorted(np.unique(labels)):
        mean_z = np.mean(trajectories[labels == cluster_id][:, :, 2], axis=0)
        plt.plot(t, mean_z, linewidth=2, label=f"Cluster {cluster_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("z (m)")
    plt.title("Mean z profile per cluster")
    plt.grid(True)
    plt.legend()
    save_path = Path(output_dir) / "cluster_z_profile.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"宸蹭繚瀛榋鏂瑰悜鏇茬嚎鍥? {save_path}")

def main():
    parser = argparse.ArgumentParser(description="鍩轰簬鏇茬巼鐨凣T杞ㄨ抗鑱氱被涓庡彲瑙嗗寲")
    parser.add_argument("--clip-index", default="clip_index.parquet")
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-end", type=int, default=179)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--output-dir", default="gt_clustering_results_av_3")
    parser.add_argument("--max-plot-per-cluster", type=int, default=200)
    args = parser.parse_args()

    clip_index = pd.read_parquet(args.clip_index)
    clip_index = clip_index[clip_index["clip_is_valid"]]
    chunk_mask = (clip_index["chunk"] >= args.chunk_start) & (clip_index["chunk"] <= args.chunk_end)
    clip_ids = clip_index[chunk_mask].index.tolist()
    logger.info(f"浠巆hunk {args.chunk_start}-{args.chunk_end} 鍏辫幏鍙?{len(clip_ids)} 涓猚lip")

    t0_offsets_us = list(range(2_000_000, 12_000_000, 1_000_000))
    logger.info(f"姣忎釜clip閲囨牱 {len(t0_offsets_us)} 涓椂鍒荤偣 (2s-10s)")

    trajectories, hist_xyz, hist_rot, fut_rot, metadata = load_gt_trajectories(clip_ids, t0_offsets_us)
    if len(trajectories) == 0:
        raise ValueError("没有加载到有效轨迹")

    logger.info("浣跨敤UnicycleAccelCurvatureActionSpace璁＄畻(鍔犻€熷害, 鏇茬巼)...")
    action_space = build_action_space()
    actions = compute_actions(action_space, trajectories, hist_xyz, hist_rot, fut_rot)
    logger.info(f"actions shape: {actions.shape}")

    logger.info(f"寮€濮嬪垎浣嶆暟鍒嗙鑱氱被 (n_clusters={args.n_clusters}, 鐗瑰緛=鏇茬巼鍧囧€?...")
    labels, kmeans, scaler, flat_norm = cluster_trajectories(actions, args.n_clusters)

    for i in range(args.n_clusters):
        count = np.sum(labels == i)
        logger.info(f"  绨?{i}: {count} 鏉¤建杩?({100 * count / len(labels):.1f}%)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    result_df = metadata.copy()
    result_df["cluster_label"] = labels
    result_df.to_csv(output_dir / "cluster_labels.csv", index=False)
    np.save(output_dir / "trajectories.npy", trajectories)
    np.save(output_dir / "actions.npy", actions)
    with open(output_dir / "cluster_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "method": "quantile_binning"}, f)
    logger.info(f"鑱氱被缁撴灉宸蹭繚瀛樺埌 {output_dir}/")

    logger.info("寮€濮嬬敓鎴愬彲瑙嗗寲鍥捐〃...")
    visualize_cluster_trajectories(trajectories, labels, output_dir, max_per_cluster=args.max_plot_per_cluster)
    visualize_cluster_centers(trajectories, labels, output_dir)
    visualize_pca_distribution(flat_norm, labels, output_dir)
    visualize_z_profile(trajectories, labels, output_dir)
    logger.info("全部可视化完成")

if __name__ == "__main__":
    main()
