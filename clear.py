import psutil
import os
import gc

pid = os.getpid()
py = psutil.Process(pid)
print(f"Memory used before cleanup: {py.memory_info().rss / 1024 ** 2} MB")

# Clear caches
gc.collect()

print(f"Memory used after cleanup: {py.memory_info().rss / 1024 ** 2} MB")