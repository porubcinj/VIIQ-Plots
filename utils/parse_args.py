import argparse

class Args(argparse.Namespace):
    def __init__(self):
        self._env: str = ".env"

    @property
    def env(self): return self._env
    @env.setter
    def env(self, value): self._env = str(value)

def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", default=Args.env)
    args = parser.parse_args(namespace=Args())
    return args