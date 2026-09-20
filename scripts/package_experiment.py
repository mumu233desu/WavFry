import os
import yaml
import json
import zipfile
from datetime import datetime
from pathlib import Path

def package_experiment():
    # 基础路径设定
    root = Path(__file__).parent.parent
    config_path = root / "config" / "_defaults.yaml"
    results_path = root / "exp_results" / "runs" / "loocv_results.json"
    
    # 1. 加载配置
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取运行目录和权重目录 (处理相对/绝对路径)
    runs_dir = Path(config['tensorboard']['log_dir'])
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir
        
    ckpt_dir = Path(config['checkpoint']['dir'])
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir

    # 2. 获取说话人数量
    spk_count = 0
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            spk_count = len(data.get('folds', []))
    
    # 3. 准备打包文件名
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"wavfry_exp_{date_str}_{spk_count}spks.zip"
    zip_path = root / zip_name

    print(f"Target: {zip_name}")
    print(f"Scanning directories...")

    # 4. 执行打包
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 打包 runs 文件夹 (日志、报表、JSON结果)
        if runs_dir.exists():
            print(f"Adding runs: {runs_dir}")
            for file in runs_dir.rglob('*'):
                if file.is_file():
                    zipf.write(file, arcname=Path('runs') / file.relative_to(runs_dir))
        else:
            print(f"Warning: Runs directory {runs_dir} not found.")

        # 打包 checkpoints 文件夹 (如果存在)
        if ckpt_dir.exists():
            print(f"Adding checkpoints: {ckpt_dir}")
            for file in ckpt_dir.rglob('*'):
                if file.is_file():
                    # 避免打包过大的 latest 文件 (可选，此处默认全部打包以保全实验状态)
                    zipf.write(file, arcname=Path('checkpoints') / file.relative_to(ckpt_dir))
        else:
            print(f"Notice: Checkpoints directory {ckpt_dir} skipped (not found).")

        # 顺便带上当前的配置文件，方便复现
        zipf.write(config_path, arcname="experiment_config.yaml")

    print(f"\nSuccessfully created: {zip_path}")
    print(f"Package Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 5. 清理文件夹 (打包成功后执行)
    import shutil
    print(f"\nCleaning up source directories...")
    if runs_dir.exists():
        try:
            shutil.rmtree(runs_dir)
            print(f"Deleted: {runs_dir}")
        except Exception as e:
            print(f"Error deleting runs: {e}")
            
    if ckpt_dir.exists():
        try:
            shutil.rmtree(ckpt_dir)
            print(f"Deleted: {ckpt_dir}")
        except Exception as e:
            print(f"Error deleting checkpoints: {e}")

if __name__ == "__main__":
    package_experiment()
