"""Backward-compatible SONIC entry point for :mod:`mocke.export.cli`."""

from mocke.export.cli import main

if __name__ == "__main__":
    main(default_recipe="sonic")
