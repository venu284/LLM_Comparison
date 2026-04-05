from __future__ import annotations

from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def count_files(path: Path, suffixes: tuple[str, ...]) -> int:
    return sum(1 for file in path.rglob("*") if file.is_file() and file.suffix in suffixes)


def count_test_files(path: Path) -> int:
    return sum(
        1
        for file in path.rglob("*")
        if file.is_file() and (".test." in file.name or ".spec." in file.name)
    )


def category_totals(root: Path, suffixes: tuple[str, ...]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)

    for file in root.rglob("*"):
        if file.is_file() and file.suffix in suffixes:
            if file.parent.name == "setup":
                continue
            totals[file.parent.name] += 1

    return dict(sorted(totals.items()))


def main() -> None:
    sections = {
        "tasks": count_files(ROOT / "tasks", (".json",)),
        "tests": count_test_files(ROOT / "tests"),
        "solutions": count_files(ROOT / "solutions", (".js", ".jsx", ".ts", ".tsx", ".css", ".html")),
    }

    print("Benchmark Summary")
    print("=================")
    for label, total in sections.items():
        print(f"{label}: {total}")

    print("\nPer-category task counts")
    print("------------------------")
    for category, total in category_totals(ROOT / "tasks", (".json",)).items():
        print(f"{category}: {total}")


if __name__ == "__main__":
    main()
