"""DTW monitor, version 3. Same nodes and edges as Lucid page 03."""
import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "SKILLS" / "diagramas-drawio-onebyte" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dgen import AMBER, BLUE, CYAN, D, GREEN, GREEN_T, TINT_B, TINT_C  # noqa: E402


def generate(out):
    d = D("dtw-monitor-v3", "DTW monitor · version 3")
    d.P("base")

    score = d.box(
        340, 80, 400, 110,
        title="Best-so-far", sub="score q(t)",
        icons=[("tag:q:#1E4FD0", 48, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    ramp = d.box(
        20, 270, 240, 120,
        title="Progress ramp", sub="reference R",
        icons=[("tag:R:#0E6E86", 48, 36)],
        fill=TINT_C, stroke="#18A9CC",
    )
    window = d.box(
        380, 270, 400, 120,
        title="Window X", sub="last W scores",
        icons=[("tag:X:#0B2252", 48, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    plateau = d.box(
        900, 270, 250, 120,
        title="Stagnation plateau", sub="reference C",
        icons=[("tag:C:#0E6E86", 48, 36)],
        fill=TINT_C, stroke="#18A9CC",
    )
    align = d.box(
        360, 560, 440, 120,
        title="DTW / DDTW", sub="trajectory alignment",
        icons=[("tag:DTW:#0B2252", 72, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    evidence = d.box(
        360, 780, 440, 110,
        title="Stagnation evidence", sub="D_R · D_C · Delta",
        icons=[("tag:EV:#B26A00", 56, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )
    policy = d.box(
        360, 1000, 440, 110,
        title="Control policy", sub="profile for t + 1",
        icons=[("tag:t+1:#1F7A45", 64, 36)],
        fill=GREEN_T, stroke=GREEN, sw=3,
    )

    d.edge(score, window, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(window, ramp, color=CYAN, dashed=True, ex=0, ey=0.5, nx=1, ny=0.5)
    d.edge(window, plateau, color=CYAN, dashed=True, ex=1, ey=0.5, nx=0, ny=0.5)
    d.edge(window, align, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(ramp, align, ex=0.55, ey=1, nx=0.08, ny=0)
    d.edge(plateau, align, ex=0.4, ey=1, nx=0.92, ny=0)
    d.text(150, 470, 40, 36, "R", size=22, color="#0E6E86", bold=True)
    d.text(560, 500, 40, 36, "X", size=22, color="#0B2252", bold=True)
    d.text(980, 470, 40, 36, "C", size=22, color="#0E6E86", bold=True)
    d.edge(align, evidence, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(evidence, policy, color=CYAN, dashed=True, label="thresholds", ex=0.5, ey=1, nx=0.5, ny=0)
    d.save(out)


if __name__ == "__main__":
    if "OUT" in globals():
        generate(OUT)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("output", nargs="?", default="figuras")
        generate(parser.parse_args().output)
