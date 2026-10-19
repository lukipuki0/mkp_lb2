"""Build a flat LaTeX source package while preserving manuscript content."""

from __future__ import annotations

import re
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ESWA_Journal_Article_Template (1)"
DESTINATION = Path(__file__).resolve().parent / "ESWA_sources_flat"
ARCHIVE = DESTINATION.with_suffix(".zip")
MAIN_SOURCE = SOURCE / "template.tex"
MAIN_DESTINATION = DESTINATION / "template.tex"

GRAPHIC_PATTERN = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^{}]+)(\})")
def replace_once(text: str, old: str, new: str, label: str) -> str:
    occurrences = text.count(old)
    if occurrences != 1:
        raise ValueError(f"Expected one {label}, found {occurrences}")
    return text.replace(old, new, 1)


def main() -> None:
    if DESTINATION.exists() or ARCHIVE.exists():
        raise FileExistsError(
            "Output already exists. Review or relocate it before rebuilding."
        )
    if not MAIN_SOURCE.is_file():
        raise FileNotFoundError(MAIN_SOURCE)

    text = MAIN_SOURCE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        r"\graphicspath{ {./figures/} }" + "\n",
        "",
        "graphics search path",
    )
    graphics = {
        path.name.lower(): path
        for path in SOURCE.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".eps"}
    }
    if len(graphics) != sum(
        1
        for path in SOURCE.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".eps"}
    ):
        raise ValueError("Two graphics have the same filename; flattening is ambiguous")

    referenced_graphics: dict[str, Path] = {}

    def flatten_graphic(match: re.Match[str]) -> str:
        original = match.group(2)
        filename = PurePosixPath(original.replace("\\", "/")).name
        source = graphics.get(filename.lower())
        if source is None:
            raise FileNotFoundError(f"Referenced graphic not found: {original}")
        referenced_graphics[filename] = source
        return match.group(1) + filename + match.group(3)

    text = GRAPHIC_PATTERN.sub(flatten_graphic, text)
    if not referenced_graphics:
        raise ValueError("No graphics were found in the manuscript")
    if re.search(r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]*[/\\]", text):
        raise ValueError("A graphic still has a directory component")

    DESTINATION.mkdir(parents=True)
    MAIN_DESTINATION.write_text(text, encoding="utf-8", newline="\n")
    supporting_files = [
        SOURCE / "elsarticle.cls",
        SOURCE / "model5-names.bst",
        SOURCE / "sample.bib",
        *sorted(referenced_graphics.values(), key=lambda path: path.name.lower()),
    ]
    for source in supporting_files:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, DESTINATION / source.name)

    with ZipFile(ARCHIVE, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(DESTINATION.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file():
                raise ValueError(f"Unexpected subfolder in output: {path}")
            archive.write(path, arcname=path.name)

    with ZipFile(ARCHIVE) as archive:
        entries = archive.namelist()
        if len(entries) != len(set(entries)):
            raise ValueError("Duplicate ZIP entries")
        if any("/" in name or "\\" in name for name in entries):
            raise ValueError("ZIP contains a directory path")
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")

    print(f"Flat source folder: {DESTINATION}")
    print(f"Flat ZIP: {ARCHIVE}")
    print(f"Referenced graphics: {len(referenced_graphics)}")
    print(f"Total ZIP entries: {len(entries)}")


if __name__ == "__main__":
    main()
