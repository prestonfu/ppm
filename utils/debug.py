import sys
import os
import pdb
import traceback


def debughook(etype, value, tb):
    traceback.print_exception(etype, value, tb)
    if True:  # not issubclass(etype, KeyboardInterrupt):
        print()  # make a new line before launching post-mortem
        pdb.pm()  # post-mortem debugger
    os._exit(0)


def enable_debug():
    sys.excepthook = debughook
