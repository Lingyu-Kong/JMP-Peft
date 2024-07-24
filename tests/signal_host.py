import os
import signal


def doit():
    os.system("brequeue -h")


def main():
    # Print current pid

    doit()

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
    doit()
    print()
    print()


if __name__ == "__main__":
    main()
