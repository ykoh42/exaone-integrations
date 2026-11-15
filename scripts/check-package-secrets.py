"""Check a built distribution for private environment files and a synthetic secret."""

import argparse
import tarfile
import zipfile
from pathlib import PurePosixPath


def read_members(path: str) -> list[tuple[str, bytes]]:
    """Read regular files from a wheel or source/npm archive."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return [
                (name, archive.read(name))
                for name in archive.namelist()
                if not name.endswith("/")
            ]
    with tarfile.open(path, "r:gz") as archive:
        return [
            (member.name, archive.extractfile(member).read())
            for member in archive
            if member.isfile()
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--expected-public-examples", type=int, required=True)
    args = parser.parse_args()

    entries = read_members(args.archive)
    public_examples = 0
    for name, content in entries:
        path = PurePosixPath(name)
        if ".direnv" in path.parts:
            raise SystemExit(f"Private direnv state included in distribution: {name}")
        basename = path.name
        if basename in {".netrc", ".npmrc", ".pypirc"}:
            raise SystemExit(f"Private authentication configuration included in distribution: {name}")
        if basename in {".env.example", ".envrc.example"}:
            public_examples += 1
        elif basename in {".env", ".envrc"} or basename.startswith((".env.", ".envrc.")):
            raise SystemExit(
                f"Private environment file included in distribution: {name}"
            )
        if b"SYNTHETIC_PACKAGE_SECRET" in content:
            raise SystemExit(f"Synthetic secret included in distribution: {name}")
    if public_examples != args.expected_public_examples:
        raise SystemExit(
            f"Expected {args.expected_public_examples} public templates, found {public_examples}"
        )
    print(
        f"{args.archive}: private environment files excluded; public templates retained"
    )


if __name__ == "__main__":
    main()
