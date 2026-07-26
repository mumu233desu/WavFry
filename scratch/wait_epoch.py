import os
import sys
import time
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Force stdout to use utf-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def wait_for_epoch():
    log_dir = Path("runs/full_ablation/full/fold_U1001")
    start_time = time.time()
    timeout = 150 # 2.5 minutes
    
    print("Waiting for Epoch 1 to finish validation and write TensorBoard metrics...")
    
    while time.time() - start_time < timeout:
        if log_dir.exists():
            files = list(log_dir.glob("events.out.tfevents.*"))
            if files:
                latest_file = max(files, key=os.path.getmtime)
                try:
                    ea = EventAccumulator(str(latest_file))
                    ea.Reload()
                    if 'Loss/val' in ea.scalars.Keys():
                        events = ea.scalars.Items('Loss/val')
                        if events:
                            print(f"\n[SUCCESS] Epoch 1 finished after {time.time() - start_time:.1f} seconds!")
                            print("Metrics:")
                            for tag in ea.scalars.Keys():
                                evs = ea.scalars.Items(tag)
                                if evs:
                                    print(f"  {tag:<30} | Step: {evs[-1].step:<5} | Value: {evs[-1].value:.6f}")
                            return
                except Exception as e:
                    pass
        print(".", end="", flush=True)
        time.sleep(10)
        
    print("\nTimeout reached before Epoch 1 finished.")

if __name__ == "__main__":
    wait_for_epoch()
