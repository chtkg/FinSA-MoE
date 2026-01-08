import os
import json
import time
import torch
from transformers import TrainerCallback

class TrainStatsCallback(TrainerCallback):
    """
    按 logging_steps 持续记录：
      - global_step
      - wall_time (elapsed_sec)
      - loss / learning_rate（来自 Trainer 的 logs）
      - GPU 峰值显存（max_memory_allocated / reserved）
    并在训练结束写 summary。
    """
    def __init__(self, output_dir: str, tag: str):
        self.output_dir = output_dir
        self.tag = tag
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_path = os.path.join(self.output_dir, f"train_stats_{tag}.jsonl")
        self.summary_path = os.path.join(self.output_dir, f"train_summary_{tag}.json")

        self.t0 = None
        self.last_step = 0

        self.peak_alloc = 0
        self.peak_reserved = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.last_step = int(state.global_step)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def _update_gpu_peak(self):
        if not torch.cuda.is_available():
            return
        # 注意：有些场景需要 synchronize 才更准
        torch.cuda.synchronize()
        alloc = int(torch.cuda.max_memory_allocated())
        rsv = int(torch.cuda.max_memory_reserved())
        self.peak_alloc = max(self.peak_alloc, alloc)
        self.peak_reserved = max(self.peak_reserved, rsv)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Trainer 每到 logging_steps 会触发 on_log，logs里一般包含 loss/lr 等
        if logs is None:
            logs = {}

        step = int(state.global_step)
        if self.t0 is None:
            self.t0 = time.time()

        self._update_gpu_peak()
        elapsed = time.time() - self.t0

        record = {
            "tag": self.tag,
            "global_step": step,
            "elapsed_sec": elapsed,
            "logs": {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))},
            "gpu_peak_alloc_bytes": self.peak_alloc,
            "gpu_peak_reserved_bytes": self.peak_reserved,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.last_step = step

    def on_train_end(self, args, state, control, **kwargs):
        if self.t0 is None:
            self.t0 = time.time()
        self._update_gpu_peak()

        summary = {
            "tag": self.tag,
            "final_global_step": int(state.global_step),
            "train_runtime_sec": float(time.time() - self.t0),
            "gpu_peak_alloc_bytes": int(self.peak_alloc),
            "gpu_peak_reserved_bytes": int(self.peak_reserved),
            "output_dir": self.output_dir,
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
