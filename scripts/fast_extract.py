#!/usr/bin/env python3
"""
Turbo Multi-Threaded Zip Extractor for 128-Core High-Memory Cloud Machines.
- Skips already extracted files instantly (checks file existence and exact size).
- Full 96-thread concurrent parallel extraction (saturates 128-thread CPU).
- Pre-caches zip archive into /dev/shm RAM disk to eliminate read/write I/O contention.
- Real-time progress bar with per-second throughput and ETA.
"""

import os
import sys
import time
import zipfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATASETS = [
    ("Corpus_CS_Dialogue.zip", "/data/Corpus/"),
    ("Corpus_Creaky_MELI.zip", "/data/Corpus/"),
    ("Corpus_Breathy_YSJ.zip", "/data/Corpus/"),
    ("Corpus_Breathy_ITA.zip", "/data/Corpus/"),
    ("Corpus_Breathy_BreathyTTS.zip", "/data/Corpus/"),
    ("Corpus_Noise_QUT.zip", "/data/Corpus/"),
    ("checkpoints_backup.zip", "/data/exp_results/"),
]

def fast_unzip_turbo(zip_path_str: str, dest_dir_str: str, workers: int = 96) -> None:
    zip_path = Path(zip_path_str)
    dest_dir = Path(dest_dir_str)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"⚠️ [跳过] 找不到文件: {zip_path}")
        return

    print(f"\n{'='*70}")
    print(f"📦 正在极速并发解压: {zip_path.name}")
    print(f"   目标目录: {dest_dir} (并发线程数: {workers})")
    print(f"{'='*70}")

    t0 = time.time()
    
    # 尝试加载到 /dev/shm 内存盘加速读取 (避免网络读 + 网络写的双重拥堵)
    use_shm = False
    shm_zip_path = Path("/dev/shm") / zip_path.name
    try:
        shm_free = shutil.disk_usage("/dev/shm").free if Path("/dev/shm").exists() else 0
        zip_size = zip_path.stat().st_size
        if shm_free > zip_size + 2 * 1024**3:
            print(f"   ⚡ 正在将压缩包预读至内存盘 /dev/shm ({zip_size / 1024 / 1024:.1f} MB)...", end="", flush=True)
            shutil.copyfile(zip_path, shm_zip_path)
            read_zip_path = shm_zip_path
            use_shm = True
            print(" 完成！")
        else:
            read_zip_path = zip_path
    except Exception:
        read_zip_path = zip_path

    with zipfile.ZipFile(read_zip_path, 'r') as zf_main:
        infolist = zf_main.infolist()
    
    total = len(infolist)
    print(f"   包内总条目: {total:,} 个")
    
    # 极速预检：找出哪些已经存在，直接秒级跳过！
    to_extract = []
    skipped = 0
    t_check = time.time()
    for info in infolist:
        target = dest_dir / info.filename
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists() and target.stat().st_size == info.file_size:
            skipped += 1
        else:
            to_extract.append(info)
    
    check_time = time.time() - t_check
    print(f"   🔍 预检完成 ({check_time:.2f}s): 已存在并跳过 {skipped:,} 个文件，剩余待解压 {len(to_extract):,} 个")

    if not to_extract:
        print(f"   ✨ 该数据集已全部解压就绪，无需重复操作！")
        if use_shm and shm_zip_path.exists():
            try:
                shm_zip_path.unlink()
            except Exception:
                pass
        return

    thread_local = threading.local()
    def get_zf():
        if not hasattr(thread_local, 'zf'):
            thread_local.zf = zipfile.ZipFile(read_zip_path, 'r')
        return thread_local.zf

    done_counter = 0
    lock = threading.Lock()
    num_to_extract = len(to_extract)

    def extract_member(info):
        nonlocal done_counter
        zf = get_zf()
        zf.extract(info, dest_dir)
        with lock:
            done_counter += 1
            if done_counter % 200 == 0 or done_counter == num_to_extract:
                elapsed = time.time() - t0
                pct = done_counter / num_to_extract * 100
                speed = done_counter / (elapsed + 1e-5)
                eta = (num_to_extract - done_counter) / (speed + 1e-5)
                print(
                    f"\r   进度: {done_counter:,}/{num_to_extract:,} ({pct:5.1f}%) | "
                    f"速度: {speed:5.0f} 文件/秒 | 剩余时间: {eta:4.1f}s",
                    end="",
                    flush=True
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(extract_member, to_extract))

    total_time = time.time() - t0
    avg_speed = num_to_extract / (total_time + 1e-5)
    print(f"\n   ✅ 完成！耗时: {total_time:.1f} 秒 (平均速度: {avg_speed:.0f} 文件/秒)")

    if use_shm and shm_zip_path.exists():
        try:
            shm_zip_path.unlink()
        except Exception:
            pass

def main():
    cpu_cores = os.cpu_count() or 32
    # 充分榨干 128 线程机器性能：开到 96 并发线程
    workers = min(96, cpu_cores)
    print(f"🚀 启动 96 线程 Turbo 极速解压 (检测到 CPU 核心数: {cpu_cores}, 并发线程数: {workers})")
    print(f"   特性: 自动秒级跳过已完成文件 + /dev/shm 内存预读 + 实时吞吐进度条\n")
    
    t_start = time.time()
    for zip_name, dest in DATASETS:
        full_path = Path("/data") / zip_name
        fast_unzip_turbo(str(full_path), dest, workers=workers)
    
    total_elapsed = time.time() - t_start
    print(f"\n🎉 全部 6 大数据集与检查点解压完成！总耗时: {total_elapsed:.1f} 秒。")

if __name__ == "__main__":
    main()
