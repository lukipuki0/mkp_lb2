"""Left column: MKP to profit. Right column: handoff into DTW. Labels in English."""
import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "SKILLS" / "diagramas-drawio-onebyte" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dgen import AMBER, BLUE, D, GREEN, GREEN_T, TINT_B, TINT_C  # noqa: E402


def generate(out):
    d = D("mkp-dtw-integracion-v3", "From MKP to DTW · version 3")
    d.text(40, 12, 520, 40, "MKP", size=24, color="#0B2252", bold=True)
    d.text(700, 12, 520, 40, "DTW", size=24, color="#0B2252", bold=True)
    d.P("base")

    instance = d.box(
        40, 70, 500, 110,
        title="Instance", sub="n items, m knapsacks",
        icons=[("tag:n:#1E4FD0", 48, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    solution = d.box(
        40, 230, 500, 110,
        title="Solution", sub="x ∈ {0,1}^n",
        icons=[("tag:x:#1E4FD0", 48, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    repair = d.box(
        40, 390, 500, 120,
        title="Repair",
        icons=[("tag:d:#0B2252", 48, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    constraint = d.box(
        40, 560, 500, 110,
        title="Constraint", sub="Rx ≤ b",
        icons=[("tag:Rx:#B26A00", 56, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )
    profit = d.box(
        40, 720, 500, 110,
        title="Profit", sub="f = p·x",
        icons=[("tag:f:#B26A00", 48, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )

    handoff = d.box(
        700, 70, 500, 110,
        title="Handoff to DTW", sub="q = f, window X",
        icons=[("tag:X:#0B2252", 48, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    ramp = d.box(
        700, 230, 230, 110,
        title="Ramp R", sub="progress",
        icons=[("tag:R:#0E6E86", 48, 36)],
        fill=TINT_C, stroke="#18A9CC",
    )
    plateau = d.box(
        970, 230, 230, 110,
        title="Plateau C", sub="constant",
        icons=[("tag:C:#0E6E86", 48, 36)],
        fill=TINT_C, stroke="#18A9CC",
    )
    dtw = d.box(
        700, 390, 500, 120,
        title="DTW / DDTW", sub="D1(X, R), D2(X, C), Delta",
        icons=[("tag:DTW:#0B2252", 64, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    rule = d.box(
        700, 560, 500, 120,
        title="Condition", sub="counter, low D2, high D1 or Delta",
        icons=[("tag:D:#B26A00", 48, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )
    stag = d.box(
        700, 730, 500, 110,
        title="Stagnation", sub="streak S_t ≥ P",
        icons=[("tag:S:#1F7A45", 48, 36)],
        fill=GREEN_T, stroke=GREEN, sw=3,
    )

    d.edge(instance, solution, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(solution, repair, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(repair, constraint, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(constraint, profit, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(profit, handoff, ex=1, ey=0.5, nx=0, ny=0.5, pts=[(620, 775), (620, 125)])
    d.edge(handoff, ramp, straight=True, ex=0.23, ey=1, nx=0.5, ny=0)
    d.edge(handoff, plateau, straight=True, ex=0.77, ey=1, nx=0.5, ny=0)
    d.edge(ramp, dtw, straight=True, ex=0.5, ey=1, nx=0.23, ny=0)
    d.edge(plateau, dtw, straight=True, ex=0.5, ey=1, nx=0.77, ny=0)
    d.edge(dtw, rule, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(rule, stag, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.save(out)


if __name__ == "__main__":
    if "OUT" in globals():
        generate(OUT)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("output", nargs="?", default="figuras/integracion-mkp-dtw")
        generate(parser.parse_args().output)
