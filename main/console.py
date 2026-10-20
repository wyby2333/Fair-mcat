"""Console output control for Fair-mcat entry points."""

import sys
from contextlib import contextmanager, redirect_stdout


class _FairConsole:
    def __init__(self, target):
        self.target = target
        self.buffer = ""

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _emit(self, text):
        normalized = text.casefold()
        if "degree" not in normalized and "pagerank" not in normalized:
            self.target.write(text)

    def write(self, text):
        self.buffer += str(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._emit(line + "\n")
        return len(text)

    def flush(self):
        if self.buffer:
            self._emit(self.buffer)
            self.buffer = ""
        self.target.flush()


@contextmanager
def fair_console():
    """Show Fair-mcat progress while hiding baseline-only rows."""
    stream = _FairConsole(sys.stdout)
    with redirect_stdout(stream):
        try:
            yield
        finally:
            stream.flush()
