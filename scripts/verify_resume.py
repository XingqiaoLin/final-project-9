"""Kill a training run mid-flight and check that restarting it is continuous.

The matrix runner leans entirely on --resume: eleven queued runs assume that a
session killed at an arbitrary moment restarts at the next epoch rather than
from zero or from a corrupt checkpoint. That assumption had never been tested
against an actual kill, only read off the code, so this script does it for real
on CPU where it costs no GPU quota.
"""

import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

REPOSITORY = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPOSITORY / "runs-resume-check"
RUN = OUTPUT_ROOT / "biosnap_random" / "full" / "seed_42"
HISTORY = RUN / "history.csv"
CHECKPOINT = RUN / "last.pt"

# The workload has to be slow enough that the poll loop can catch the run
# between epochs. With 64 training pairs an epoch finished faster than the
# polling interval and the "interruption" landed after the run was already
# done, which passed every check while testing nothing.
EPOCHS = 6
KILL_AFTER_EPOCHS = 2
TRAIN_PAIRS = 1500

COMMAND = [
    sys.executable,
    "train_e3dti.py",
    "--dataset", "biosnap", "--split", "random", "--seed", "42",
    "--epochs", str(EPOCHS),
    "--batch", "4", "--dim", "32",
    "--drug-layers", "1", "--protein-layers", "1",
    "--max-train-pairs", str(TRAIN_PAIRS), "--max-val-pairs", "256", "--max-test-pairs", "256",
    "--device", "cpu",
    "--output-root", str(OUTPUT_ROOT),
    "--resume",
]


def completed_epochs():
    """Epochs finished so far, read from the checkpoint.

    history.csv is only written once, after the epoch loop, so it is useless as
    a progress signal while a run is in flight -- polling it meant the "kill"
    always landed after the run had already finished. last.pt is rewritten
    every epoch and carries the same history inside it.
    """
    if not CHECKPOINT.exists():
        return []
    try:
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    except Exception:
        # A read that races the atomic rename is expected; try again next poll.
        return []
    return [str(row["epoch"]) for row in state["history"]]


def final_epochs():
    if not HISTORY.exists():
        return []
    with HISTORY.open() as handle:
        return [row["epoch"] for row in csv.DictReader(handle)]


def launch():
    # New process group, so the kill reaches the trainer and not just a shell.
    return subprocess.Popen(
        COMMAND,
        cwd=REPOSITORY,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def main():
    if OUTPUT_ROOT.exists():
        subprocess.run(["rm", "-rf", str(OUTPUT_ROOT)], check=True)

    print(f"launching, will SIGKILL after {KILL_AFTER_EPOCHS} epochs land")
    process = launch()
    deadline = time.time() + 600

    while True:
        if time.time() > deadline:
            process.kill()
            print("FAIL: timed out before the kill point")
            return 1
        if process.poll() is not None:
            print("FAIL: the run finished before we could interrupt it")
            return 1
        if len(completed_epochs()) >= KILL_AFTER_EPOCHS:
            # SIGKILL, not SIGTERM: a session death gives no chance to clean up,
            # so anything that relies on a shutdown hook must not pass this.
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            break
        time.sleep(0.2)

    process.wait()
    before = completed_epochs()
    print(f"killed with epochs {before} on disk")

    checkpoint = CHECKPOINT
    if not checkpoint.exists():
        print("FAIL: no last.pt was written, nothing to resume from")
        return 1
    print(f"last.pt is {checkpoint.stat().st_size / 1e6:.1f} MB")

    print("relaunching the identical command")
    second = subprocess.run(COMMAND, cwd=REPOSITORY, capture_output=True, text=True)
    if second.returncode != 0:
        print("FAIL: the resumed run exited", second.returncode)
        print(second.stderr[-3000:])
        return 1

    after = final_epochs()
    print(f"epochs after resume: {after}")

    expected = [str(n) for n in range(1, EPOCHS + 1)]
    problems = []
    # Without this the test passes trivially when the run outpaces the poll
    # loop and the "kill" lands after the final epoch.
    if len(before) >= EPOCHS:
        problems.append(
            f"the kill landed after all {EPOCHS} epochs had finished, so no "
            f"interruption was tested"
        )
    if after != expected:
        problems.append(f"history is {after}, expected {expected}")
    if len(set(after)) != len(after):
        problems.append("an epoch was recomputed and logged twice")
    if before != after[: len(before)]:
        problems.append("the resumed run rewrote the pre-kill history")
    if not (RUN / "results.json").exists():
        problems.append("no results.json, the run did not actually finish")

    if problems:
        for problem in problems:
            print("FAIL:", problem)
        return 1

    print(
        f"\nPASS: killed after epoch {len(before)}, resumed at "
        f"{len(before) + 1}, finished at {EPOCHS}, with no epoch recomputed "
        f"and no history rewritten."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
