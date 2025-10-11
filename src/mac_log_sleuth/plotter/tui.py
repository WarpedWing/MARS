#!/usr/bin/env python3

"""
ANSI color codes and styled output functions.
"""


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    IT = "\033[3m"
    U = "\033[4m"

    FG = type(
        "FG",
        (),
        {
            "GRAY": "\033[90m",
            "RED": "\033[31m",
            "GREEN": "\033[32m",
            "YELLOW": "\033[33m",
            "BLUE": "\033[34m",
            "MAGENTA": "\033[35m",
            "CYAN": "\033[36m",
            "WHITE": "\033[37m",
        },
    )


def cinfo(msg):
    print(f"{C.FG.CYAN}{msg}{C.RESET}")


def cgood(msg):
    print(f"{C.FG.GREEN}{msg}{C.RESET}")


def cwarn(msg):
    print(f"{C.FG.YELLOW}{msg}{C.RESET}")


def cerr(msg):
    print(f"{C.FG.RED}{msg}{C.RESET}")


def chead(msg):
    print(f"{C.BOLD}{msg}{C.RESET}")
