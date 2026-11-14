"""DTW monitor, version 3. Same nodes and edges, folded into a squarer layout."""
import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "SKILLS" / "diagramas-drawio-onebyte" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dgen import AMBER, BLUE, CYAN, D, GREEN, GREEN_T, TINT_B, TINT_C  # noqa: E402


def generate(out):
    d = D("dtw-monitor-v3", "DTW monitor · version 3")
    d.P("base")

    score = d.box(
        40, 30, 360, 110,
        title="Best-so-far", sub="score q(t)",
        icons=[("tag:q:#1E4FD0", 48, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    window = d.box(
        40, 220, 360, 110,
        title="Window X", sub="last W scores",
        icons=[("tag:X:#0B2252", 48, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    ramp = d.box(
        480, 30, 380, 110,
        title="Progress ramp", sub="reference R",
        icons=[("tag:R:#0E6E86", 48, 36)],
        fill=TINT_C, stroke="#18A9CC",
    )
    align = d.box(
        480, 220, 380, 110,
        title="DTW / DDTW", sub="trajectory alignment",
        icons=[("tag:DTW:#0B2252", 72, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    plateau = d.box(
        480, 410, 380, 110,
        title="Stagnation plateau", sub="reference C",
        icons=[("tag:C:#0E6E86", 48, 36)],
        fill=TINT_C, stroke="#18A9CC",
    )
    evidence = d.box(
        940, 220, 420, 110,
        title="Stagnation evidence", sub="D_R · D_C · Delta",
        icons=[("tag:EV:#B26A00", 56, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )
    policy = d.box(
        940, 410, 420, 110,
        title="Control policy", sub="profile for t + 1",
        icons=[("tag:t+1:#1F7A45", 64, 36)],
        fill=GREEN_T, stroke=GREEN, sw=3,
    )

    d.edge(score, window, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(window, ramp, color=CYAN, dashed=True, straight=True, ex=1, ey=0.25, nx=0, ny=0.75)
    d.edge(window, plateau, color=CYAN, dashed=True, straight=True, ex=1, ey=0.75, nx=0, ny=0.25)
    d.edge(window, align, straight=True, ex=1, ey=0.5, nx=0, ny=0.5)
    d.edge(ramp, align, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(plateau, align, straight=True, ex=0.5, ey=0, nx=0.5, ny=1)
    d.text(430, 188, 36, 32, "X", size=22, color="#0B2252", bold=True)
    d.text(640, 168, 36, 32, "R", size=22, color="#0E6E86", bold=True)
    d.text(640, 348, 36, 32, "C", size=22, color="#0E6E86", bold=True)
    d.edge(align, evidence, straight=True, ex=1, ey=0.5, nx=0, ny=0.5)
    d.edge(evidence, policy, color=CYAN, dashed=True, straight=True, ex=0.62, ey=1, nx=0.62, ny=0)
    d.text(1228, 344, 180, 36, "thresholds", size=22, color="#2B3F66", bold=True)
    d.save(out)


if __name__ == "__main__":
    if "OUT" in globals():
        generate(OUT)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("output", nargs="?", default="figuras")
        generate(parser.parse_args().output)
