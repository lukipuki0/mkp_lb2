"""HRES2 energy flow on the DTW monitor grid. Same nodes and edges."""
import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "SKILLS" / "diagramas-drawio-onebyte" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dgen import AMBER, BLUE, D, GREEN, GREEN_T, TINT_B, TINT_C  # noqa: E402


def generate(out):
    d = D("hres2-flujo-energetico-v3", "HRES2 energy flow · version 3")
    d.P("base")

    # Tight 3-column flow. Wind and solar share width and x, mirrored
    # about renewable generation. BESS is wide enough for one subtitle line.
    # Green H2 sits under the PEM electrolyzer; grid sits under generation.
    eolica = d.box(
        16, 51, 400, 110,
        title="Wind", sub="P<sub>WT</sub>",
        icons=[("tag:WT:#1E4FD0", 48, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    solar = d.box(
        16, 321, 400, 110,
        title="Solar PV", sub="P<sub>PV</sub> = 200 − P<sub>WT</sub>",
        icons=[("tag:PV:#1E4FD0", 48, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    bess = d.box(
        825, 16, 350, 120,
        title="BESS", sub="stores surplus",
        icons=[("tag:BAT:#B26A00", 56, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )
    gen = d.box(
        464, 181, 400, 120,
        title="Renewable generation",
        icons=[("tag:GEN:#0B2252", 64, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    red = d.box(
        539, 346, 250, 120,
        title="Grid", sub="sells surplus",
        icons=[("tag:GRID:#0E6E86", 64, 36)],
        fill=TINT_C, stroke="#18A9CC",
    )
    el = d.box(
        912, 181, 440, 120,
        title="PEM electrolyzer", sub="P<sub>el</sub> = 5N<sub>el</sub>",
        icons=[("tag:PEM:#B26A00", 64, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    h2 = d.box(
        912, 351, 440, 110,
        title="Green H<sub>2</sub>", sub="final product",
        icons=[("tag:H2:#1F7A45", 48, 36)],
        fill=GREEN_T, stroke=GREEN, sw=3,
    )

    d.edge(eolica, gen, straight=True, ex=1, ey=0.5, nx=0, ny=0.25)
    d.edge(solar, gen, straight=True, ex=1, ey=0.5, nx=0, ny=0.75)
    d.edge(gen, bess, straight=True, ex=0.82, ey=0, nx=0.25, ny=1)
    d.edge(gen, red, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(gen, el, straight=True, ex=1, ey=0.5, nx=0, ny=0.5)
    d.edge(bess, el, straight=True, ex=0.5, ey=1, nx=0.2, ny=0)
    d.edge(el, h2, color=GREEN, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.save(out)


if __name__ == "__main__":
    if "OUT" in globals():
        generate(OUT)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("output", nargs="?", default="figuras")
        generate(parser.parse_args().output)
