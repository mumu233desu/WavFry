import sys
import yaml
import json
import pandas as pd
from pathlib import Path

# Add paths to allow importing from the creaky tools pipeline
workspace_root = Path(__file__).resolve().parents[2]
tool_root = workspace_root / "04_creaky_tools"
sys.path.insert(0, str(tool_root / "evaluation/creaky_detection_algo_improve/scripts"))
import run_pipeline_v2 as pipe

def main():
    # 1. Load the list of all annotated speakers from the WavFry config
    wavfry_cfg_path = workspace_root / "06_wavfry/config/_defaults.yaml"
    with open(wavfry_cfg_path, 'r', encoding='utf-8') as f:
        wavfry_cfg = yaml.safe_load(f)
        
    annotated_speakers = wavfry_cfg['data']['annotated_speakers']
    print(f"Annotated speakers ({len(annotated_speakers)}): {annotated_speakers}")
    
    # 2. Load the evaluation pipeline config to get paths and external detector settings
    pipeline_cfg_path = tool_root / "evaluation/creaky_detection_algo_improve/configs/full_pipeline_config.json"
    pipeline_cfg = pipe.load_pipeline_config(pipeline_cfg_path)
    
    # Ensure deepfry and creapy are enabled for this run
    external_cfg = pipeline_cfg.get("task_e_external", {})
    if "deepfry" not in external_cfg: external_cfg["deepfry"] = {}
    external_cfg["deepfry"]["enabled"] = True
    if "creapy" not in external_cfg: external_cfg["creapy"] = {}
    external_cfg["creapy"]["enabled"] = True
    
    # 3. Resolve paths relative to workspace
    audio_dir = pipe.resolve_path(workspace_root, pipeline_cfg["audio_dir"])
    textgrid_dir = pipe.resolve_path(workspace_root, pipeline_cfg["textgrid_dir"])
    f0_methods = pipeline_cfg["f0_methods"]
    
    print("Building record maps...")
    record_maps = pipe.build_record_maps(workspace_root, audio_dir, textgrid_dir, f0_methods)
    rmvpe_map = record_maps.get("rmvpe", {})
    
    # 4. Load the full annotation set
    ann_path = pipe.resolve_path(workspace_root, pipeline_cfg["manual_annotation_file"])
    annotations = pipe.load_annotations(ann_path)
    
    # 5. Build validation manifest containing all annotated data
    stage_root = workspace_root / "06_wavfry/runs/baselines_eval"
    stage_root.mkdir(parents=True, exist_ok=True)
    
    print("Building validation manifest for full data...")
    val_df, summary = pipe.build_val_manifest(
        annotation_payload=annotations,
        target_speakers=annotated_speakers,
        record_map=rmvpe_map,
        tier=pipeline_cfg["tier"],
        vowel_phonemes=pipeline_cfg["vowel_phonemes"],
        silence_regex=pipeline_cfg["silence_regex"],
        results_dir=stage_root,
        reports_dir=stage_root,
        language_filter=None,
    )
    
    print(f"Validation dataset: {len(val_df)} segments")
    
    # 6. Evaluate DeepFry and CreaPy
    val_file_keys = sorted(val_df["file_key"].unique().tolist())
    
    deepfry_configs = {
        "deepfry_v1_noann": ("04_creaky_tools/DeepFry/models/CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth", True),
        "deepfry_v1_withann": ("04_creaky_tools/DeepFry/models/CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth", False),
        "deepfry_v2_noann": ("04_creaky_tools/DeepFry/models/CREAK-74lr_0.001_decay_38_input_size_128_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_logtxt_ff1_.pth", True),
        "deepfry_v2_withann": ("04_creaky_tools/DeepFry/models/CREAK-74lr_0.001_decay_38_input_size_128_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_logtxt_ff1_.pth", False),
    }
    
    results = []
    
    for name, (m_path, is_custom) in deepfry_configs.items():
        print(f"Running {name}...")
        pred, status = pipe.run_deepfry_external_variant(
            workspace_root=workspace_root,
            deepfry_cfg=external_cfg["deepfry"],
            variant_name=name,
            model_rel_path=m_path,
            is_custom=is_custom,
            stage_root=stage_root,
            val_file_keys=val_file_keys,
            val_df=val_df,
            rmvpe_record_map=rmvpe_map,
        )
        if status["ok"]:
            df_eval = pipe.evaluate_prediction_map(val_df, pred, name)
            results.append(df_eval)
        else:
            print(f"Failed {name}: {status.get('error')}")
            
    print("Running creapy...")
    pred, status = pipe.run_creapy_external(
        workspace_root=workspace_root,
        external_cfg=external_cfg,
        stage_root=stage_root,
        val_file_keys=val_file_keys,
        val_df=val_df,
        rmvpe_record_map=rmvpe_map,
    )
    if status["ok"]:
        df_eval = pipe.evaluate_prediction_map(val_df, pred, "creapy")
        results.append(df_eval)
    else:
        print(f"Failed creapy: {status.get('error')}")
        
    if not results:
        print("No models were successfully evaluated. Check the console errors.")
        return
        
    all_eval_df = pd.concat(results, ignore_index=True)
    csv_path = stage_root / "baselines_all_data.csv"
    all_eval_df.to_csv(csv_path, index=False)
    
    # 7. Generate Markdown Report
    md_path = stage_root / "baselines_per_speaker_report.md"
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Baselines Evaluation on All Annotated Data\n\n")
        
        models = all_eval_df['system'].unique()
        
        for model in models:
            f.write(f"## Model: {model}\n\n")
            model_df = all_eval_df[(all_eval_df['system'] == model) & (all_eval_df['split'] == 'speaker')]
            
            # Overall row
            overall_rows = all_eval_df[(all_eval_df['system'] == model) & (all_eval_df['split'] == 'overall')]
            if not overall_rows.empty:
                overall_row = overall_rows.iloc[0]
                f.write(f"**Overall**: Accuracy={overall_row['accuracy']:.4f}, Precision={overall_row['precision']:.4f}, Recall={overall_row['recall']:.4f}, F1={overall_row['f1']:.4f}\n\n")
            
            f.write("| Speaker | Accuracy | Precision | Recall | F1 |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for _, row in model_df.iterrows():
                f.write(f"| {row['group']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |\n")
                
            f.write("\n")
            
    print(f"Done! Report saved to {md_path}")
    print(f"Raw CSV data saved to {csv_path}")

if __name__ == "__main__":
    main()
