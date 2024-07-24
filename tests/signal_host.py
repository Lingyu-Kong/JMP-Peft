import os
import signal

import rich


def print_env():
    print("Environment:")
    rich.print(list(os.environ.keys()))
    os.system("which python")


def main():
    # Print current pid

    print_env()

    print("PID: ", os.getpid())

    signal.signal(signal.SIGUSR1, usr1_handler)

    print("Press any key to exit.")
    input()


def usr1_handler(signum, frame):
    print()
    print()
    print()
    print(f"Received SIGUSR1 signal. signum: {signum}, frame: {frame}")
    print()
    print()
    print_env()
    print()
    print()


if __name__ == "__main__":
    main()
