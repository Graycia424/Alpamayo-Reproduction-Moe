# -*- coding: utf-8 -*-
"""
本文件属于 Alpamayo 项目。
主要功能：多 GPU 并行运行包装器。
通过拦截 `--chunk-ids` 和 `--devices` 参数，将不同的数据块分配到不同的 GPU 显卡上，
调用底层的 `run_multiple_clips.py` 脚本执行并行推理，
并在所有进程完成后自动合并评测的 CSV 性能结果（如平均 ADE）。
"""

import os
import subprocess
import sys
import argparse

def parse_args():
    # 我们拦截 --chunk-ids 和 --devices 这两个需要用来分派任务的参数
    # 如果用户没有写 --devices，就把其它所有参数继续透传给底层的脚本
    p = argparse.ArgumentParser(description="多 GPU 并行运行包装器")
    p.add_argument("--chunk-ids", type=str, default="10,20,30,40,50,60,70,80,90,100", help="Comma-separated chunk ids")
    p.add_argument("--devices", type=str, default="cuda:0,cuda:1", help="逗号分隔的设备，如 cuda:0,cuda:1")
    return p.parse_known_args()

def main():
    args, unknown_args = parse_args()
    
    chunks = [c.strip() for c in args.chunk_ids.split(",") if c.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    
    if len(devices) == 0:
        print("未指定设备！")
        return
        
    num_devices = len(devices)
    
    # 将 chunk 分发到不同的设备上
    gpu_chunks = {i: [] for i in range(num_devices)}
    for i, chunk in enumerate(chunks):
        gpu_chunks[i % num_devices].append(chunk)
        
    processes = []
    
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = f"outputs/parallel_run_{timestamp}"
    
    for i in range(num_devices):
        device = devices[i]
        assigned_chunks = gpu_chunks[i]
        
        if not assigned_chunks:
            continue
            
        chunk_str = ",".join(assigned_chunks)
        print(f"=>{device} 被分配了 chunks: {chunk_str}")
        
        # 为了避免多个进程写入同一个文件导致冲突，我们给每个显卡单独命名 CSV
        safe_device_name = device.replace(":", "")
        csv_name = f"{base_out}/results_{safe_device_name}.csv"
        
        # 组装底层运行的命令
        cmd = [
            sys.executable, "run_multiple_clips.py",
            "--chunk-ids", chunk_str,
            "--device", device,
            "--exp-dir", base_out,
            "--results-csv", csv_name,
            "--no-add-timestamp"
        ] + unknown_args
        
        # 启动非阻塞的子进程
        p = subprocess.Popen(cmd)
        processes.append((device, p))
        
    print(f"\n🚀 已启动 {len(processes)} 个并行进程，正在满载运行！按 Ctrl+C 可以强制终止所有进程。\n")
    
    # 等待所有子进程完成
    try:
        for device, p in processes:
            p.wait()
            if p.returncode == 0:
                print(f"✅ {device} 上的任务已成功完成！")
            else:
                print(f"❌ {device} 上的任务出现错误结束，Code: {p.returncode}")
                
        # 所有进程结束后，自动帮你合并 CSV 并计算平均值！
        print("\n" + "="*50)
        print("🎉 正在合并并计算所有显卡的成绩...")
        print("="*50)
        
        all_rows = []
        header = None
        import glob
        csv_files = glob.glob(f"{base_out}/results_*.csv")
        
        for csv_file in csv_files:
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if not lines:
                        continue
                    if header is None:
                        header = lines[0] # 保存第一行的表头
                    all_rows.extend(lines[1:]) # 把其余的内容加进去
            except Exception as e:
                print(f"⚠️ 无法读取 {csv_file}: {e}")
                
        if all_rows:
            merged_csv_path = f"{base_out}/results_merged_all.csv"
            with open(merged_csv_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.writelines(all_rows)
            
            # 计算 Average ADE
            ade_values = []
            for row in all_rows:
                parts = row.strip().split(",")
                if len(parts) >= 2:
                    try:
                        ade_values.append(float(parts[1]))
                    except ValueError:
                        pass
                        
            if ade_values:
                mean_ade = sum(ade_values) / len(ade_values)
                msg = (
                    f"✅ 成功合并了 {len(csv_files)} 个独立的结果文件 -> {merged_csv_path}\n"
                    f"📊 总计读取到了 {len(ade_values)} 个 clip 的评估结果。\n"
                    f"🏆 【完整测试集平均成绩】 Mean ADE: {mean_ade:.4f} 米"
                )
                print(msg)
                
                # 同步记录到 log 文件
                log_file = f"{base_out}/run.log"
                with open(log_file, "a", encoding="utf-8") as log_f:
                    log_f.write("\n" + "="*50 + "\n")
                    log_f.write("🎉 最终测试集平均成绩汇总\n")
                    log_f.write("="*50 + "\n")
                    log_f.write(msg + "\n")
                    log_f.write("="*50 + "\n")
            else:
                print("⚠️ 合并后的表格里没有找到有效的评分数据。")
        else:
            print("⚠️ 未找到任何生成的 csv 结果文件。")
            
        print("="*50 + "\n")

    except KeyboardInterrupt:
        print("\n收到中断信号，正在终止所有子进程...")
        for _, p in processes:
            p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
