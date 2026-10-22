import ctypes
import time
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

print("[KeepAwake] Setting thread execution state to prevent sleep...", flush=True)
res = ctypes.windll.kernel32.SetThreadExecutionState(
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
)
print(f"[KeepAwake] SetThreadExecutionState returned: {res}. System sleep is locked.", flush=True)

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("[KeepAwake] Exiting cleanly.", flush=True)
