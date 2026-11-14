"""Left column: HRES2 to LCOE. Right column: handoff into DTW."""
import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "SKILLS" / "diagramas-drawio-onebyte" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dgen import AMBER, BLUE, D, GREEN, GREEN_T, TINT_B, TINT_C  # noqa: E402


def generate(out):
    d = D("hres2-dtw-integracion-v3", "From HRES2 to DTW · version 3")
    d.text(40, 12, 520, 40, "HRES2", size=24, color="#0B2252", bold=True)
    d.text(700, 12, 520, 40, "DTW", size=24, color="#0B2252", bold=True)
    d.P("base")

    design = d.box(
        40, 70, 500, 110,
        title="Design", sub="P_WT, N_el, P_bat, tau_bat",
        icons=[("tag:x:#1E4FD0", 48, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    gen = d.box(
        40, 230, 500, 110,
        title="Generation", sub="P_PV = 200 − P_WT",
        icons=[("tag:GEN:#1E4FD0", 64, 36)],
        fill=TINT_B, stroke=BLUE,
    )
    sim = d.box(
        40, 390, 500, 120,
        title="Hourly simulation", sub="8760 h, PEM and BESS dispatch",
        icons=[("tag:8760:#0B2252", 72, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    agsr = d.box(
        40, 560, 500, 110,
        title="Constraint", sub="AGSR ≤ 0.20",
        icons=[("tag:AGSR:#B26A00", 72, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )
    lcoe = d.box(
        40, 720, 500, 110,
        title="LCOE", sub="feasible f, or penalty",
        icons=[("tag:f:#B26A00", 48, 36)],
        fill="#FFF3DC", stroke=AMBER,
    )

    handoff = d.box(
        700, 70, 500, 110,
        title="Handoff to DTW", sub="q = −f, window X",
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

    d.edge(design, gen, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(gen, sim, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(sim, agsr, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(agsr, lcoe, straight=True, ex=0.5, ey=1, nx=0.5, ny=0)
    d.edge(lcoe, handoff, ex=1, ey=0.5, nx=0, ny=0.5, pts=[(620, 775), (620, 125)])
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
        parser.add_argument("output", nargs="?", default="figuras")
        generate(parser.parse_args().output)
