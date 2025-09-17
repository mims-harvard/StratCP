"""Entry point when invoked with python -m src."""  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
    import sys

    from src.cli import cli

    if sys.argv[0].endswith("__main__.py"):
        sys.argv[0] = "python -m src"
    cli()
