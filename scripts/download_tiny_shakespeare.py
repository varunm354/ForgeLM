"""Download the Tiny Shakespeare corpus."""

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DEFAULT_DESTINATION = Path("data/raw/tiny_shakespeare.txt")


def download(destination: Path, force: bool = False) -> bool:
    """Download the corpus, returning whether a file was written."""
    if destination.exists() and not force:
        print(f"Already exists, skipping: {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Tiny Shakespeare from {SOURCE_URL}")
    try:
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            contents = response.read()
        destination.write_bytes(contents)
    except (OSError, urllib.error.URLError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return False

    print(f"Saved {len(contents):,} bytes to {destination}")
    return True


def main() -> int:
    """Parse arguments and download Tiny Shakespeare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the destination if it already exists",
    )
    args = parser.parse_args()

    existed = DEFAULT_DESTINATION.exists()
    downloaded = download(DEFAULT_DESTINATION, force=args.force)
    if downloaded or (existed and not args.force):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
