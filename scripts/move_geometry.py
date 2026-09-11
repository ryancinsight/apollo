"""Extract the four complete CSV blocks from a native move-geometry run."""

import argparse
from pathlib import Path


def extract(log: Path, destination: Path) -> None:
    blocks: list[list[str]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("MOVE-GEOMETRY block="):
            expected = f"MOVE-GEOMETRY block={len(blocks)};"
            if not line.startswith(expected):
                raise ValueError(f"unexpected block order: {line}")
            blocks.append([])
        elif line.startswith(("case,min_ps,", "geometry/")):
            if not blocks:
                raise ValueError("CSV record precedes its block marker")
            blocks[-1].append(line)
    if len(blocks) != 4 or any(len(block) != 14 for block in blocks):
        raise ValueError("expected four headers and thirteen records per block")
    cases = {line.split(",", 1)[0] for line in blocks[0][1:]}
    if len(cases) != 13 or any(
        {line.split(",", 1)[0] for line in block[1:]} != cases
        for block in blocks
    ):
        raise ValueError("block case sets differ or contain duplicates")
    outputs = [(destination / f"block-{i}.csv", "\n".join(block) + "\n")
               for i, block in enumerate(blocks)]
    for path, content in outputs:
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite different evidence: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    for path, content in outputs:
        path.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    extract(arguments.log, arguments.destination)
