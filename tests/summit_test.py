# %%
import datetime
import logging
import os
import re
import signal
import sys
from pathlib import Path

import nshrunner as nr

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def requeue():
    # Find job id
    if (job_id := os.getenv("LSB_JOBID")) is None:
        log.warning("LSB_JOBID environment variable not found. Unable to requeue job.")
        return

    assert re.match("[0-9_-]+", job_id)

    exe = "brequeue"
    if (bin_dir := os.getenv("LSF_BINDIR")) is not None:
        exe = str((Path(bin_dir) / exe).resolve().absolute())

    log.info(f"Using LSF requeue executable: {exe}")

    # If NSHRUNNER_LSF_EXIT_SCRIPT_DIR exists, we should emit a bash script in that directory
    # rather than calling the requeue command directly. This is because the requeue command
    # is only available outside of the `jsrun` context, and the exit script is called within
    # the `jsrun` context.
    if not (exit_script_dir := os.getenv("NSHRUNNER_LSF_EXIT_SCRIPT_DIR")):
        raise NotImplementedError(
            "NSHRUNNER_LSF_EXIT_SCRIPT_DIR environment variable not found. Unable to requeue job."
        )

    log.critical(
        "Environment variable NSHRUNNER_LSF_EXIT_SCRIPT_DIR found.\n"
        "Writing requeue script to exit script directory."
    )
    exit_script_dir = Path(exit_script_dir)
    assert (
        exit_script_dir.is_dir()
    ), f"Exit script directory {exit_script_dir} does not exist"

    exit_script_path = exit_script_dir / f"requeue_{job_id}.sh"
    log.info(f"Writing requeue script to {exit_script_path}")

    with exit_script_path.open("w") as f:
        f.write(f"#!/bin/bash\n{exe} {job_id}\n")

    # Make the script executable
    os.chmod(exit_script_path, 0o755)

    log.info(f"Requeue script written to {exit_script_path}")


def run_fn():
    # Print current pid

    def _handler(signum, frame):
        print()
        print()
        print()
        print(f"Received SIGU signal. signum: {signum}, frame: {frame}")
        print()
        print()
        requeue()
        print()
        print()

    print("PID: ", os.getpid())

    signal.signal(signal.SIGURG, _handler)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    while True:
        pass


runner = nr.Runner(run_fn)
runner.submit_lsf(
    [()],
    {
        "summit": True,
        "project": "MAT273",
        "queue": "debug",
        "nodes": 1,
        "rs_per_node": 1,
        "walltime": datetime.timedelta(minutes=10),
    },
    snapshot=True,
    env={
        "LL_DISABLE_TYPECHECKING": "1",
    },
)
