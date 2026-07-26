import os
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Force stdout to use utf-8 to avoid encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def check_progress():
    log_dir = Path("runs/full_ablation/full/fold_U1001")
    if not log_dir.exists():
        print(f"Log directory '{log_dir}' does not exist yet. Training might be preparing or starting Epoch 1...")
        return
        
    files = list(log_dir.glob("events.out.tfevents.*"))
    if not files:
        print("No tfevents files found yet.")
        return
        
    # Get the latest file by modification time
    latest_file = max(files, key=os.path.getmtime)
    print(f"Reading from: {latest_file}")
    
    # Load events
    ea = EventAccumulator(str(latest_file))
    ea.Reload()
    
    # Print metrics
    tags = ea.Tags()
    print("\nAvailable Tags:")
    for category, tag_list in tags.items():
        if tag_list:
            safe_list = [str(x).encode('ascii', 'replace').decode('ascii') for x in tag_list]
            print(f"  {category}: {safe_list}")
            
    print("\nLatest Scalar Values:")
    for tag in ea.scalars.Keys():
        events = ea.scalars.Items(tag)
        if events:
            last_event = events[-1]
            print(f"  {tag:<30} | Step: {last_event.step:<5} | Value: {last_event.value:.6f}")

if __name__ == "__main__":
    check_progress()
