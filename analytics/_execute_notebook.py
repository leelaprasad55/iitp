import asyncio
from pathlib import Path

import nbformat
from nbclient import NotebookClient

try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass


def run(nb_path: Path) -> None:
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent.resolve())}},
    )
    client.execute()
    nbformat.write(nb, nb_path)
    print("Executed", nb_path)


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1]))
