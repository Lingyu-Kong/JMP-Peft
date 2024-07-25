# %%
import datetime
import os
import signal
import sys

import nshrunner as nr


def run_fn():
    # Print current pid

    def _handler(signum, frame):
        print()
        print()
        print()
        print(f"Received SIGU signal. signum: {signum}, frame: {frame}")
        print()
        print()
        os.system("brequeue -h")
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
