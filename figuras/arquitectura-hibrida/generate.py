"""Hybrid architecture, version 3. Exact labels and directed relationships."""
import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "SKILLS" / "diagramas-drawio-onebyte" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dgen import BLUE, D, TINT_B, TINT_C  # noqa: E402


def generate(out):
    d = D("GWODE-ArquitecturaHibrida-v3", "Hybrid architecture - version 3")
    d.P("p1", "Parte 1 · Hybrid operators")
    population = d.box(
        110, 60, 870, 110,
        title="Shared population", sub="continuous vector, binary solution, fitness",
        icons=[("tag:Pop:#1E4FD0", 72, 36)],
        fill=TINT_B, stroke=BLUE, sw=3,
    )
    alpha = d.box(
        220, 270, 220, 90,
        title="Alpha",
        icons=[("tag:A:#1E4FD0", 42, 36)],
    )
    beta = d.box(
        490, 270, 220, 90,
        title="Beta",
        icons=[("tag:B:#1E4FD0", 42, 36)],
    )
    delta = d.box(
        760, 270, 220, 90,
        title="Delta",
        icons=[("tag:D:#1E4FD0", 42, 36)],
    )
    d.text(400, 210, 180, 36, "copied, frozen", align="center")
    d.edge(population, alpha, ex=0.25287356321839083, ey=1, nx=0.5, ny=0, straight=True)
    d.edge(population, beta, ex=0.5632183908045977, ey=1, nx=0.5, ny=0, straight=True)
    d.edge(population, delta, ex=0.8735632183908046, ey=1, nx=0.5, ny=0, straight=True)

    d.P("p2", "Parte 2 · Monitor and profile")
    monitor = d.box(
        1180, 60, 360, 110,
        title="Monitor",
        icons=[("tag:Mon:#0E6E86", 72, 36)],
        fill=TINT_C, stroke="#18A9CC", sw=3,
    )
    profile = d.box(
        1180, 270, 360, 110,
        title="Profile", sub="a, F, CR",
        icons=[("tag:Prof:#0B2252", 72, 36)],
    )
    d.text(1030, 70, 100, 36, "curve", align="center")
    d.text(1220, 190, 120, 36, "profile", align="right")
    d.edge(population, monitor, dashed=True, ex=1, ey=0.5, nx=0, ny=0.5, straight=True)
    d.edge(monitor, profile, ex=0.5, ey=1, nx=0.5, ny=0, straight=True)

    d.P("p1")
    gwo = d.box(
        355, 500, 320, 110,
        title="GWO", sub="parameter a",
        icons=[("tag:GWO:#0B2252", 72, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    de = d.box(
        760, 500, 260, 110,
        title="DE", sub="F, CR",
        icons=[("tag:DE:#0B2252", 62, 36)],
        fill="#FFFFFF", stroke=BLUE, sw=3,
    )
    d.text(500, 390, 100, 36, "guided", align="center")
    d.text(980, 360, 130, 36, "individual i", align="right")
    d.edge(alpha, gwo, ex=1, ey=0.5, nx=0.16, ny=0,
           pts=[(480, 315), (480, 460), (406, 460)], straight=True)
    d.edge(beta, gwo, ex=0.5, ey=1, nx=0.5, ny=0,
           pts=[(600, 430), (515, 430)], straight=True)
    d.edge(delta, gwo, ex=0.5, ey=1, nx=0.84, ny=0,
           pts=[(870, 430), (624, 430)], straight=True)
    d.edge(population, de, ex=1, ey=0.5, nx=1, ny=0.5,
           pts=[(1120, 115), (1120, 555)], straight=True)

    d.P("p2")
    d.edge(profile, gwo, dashed=True, ex=1, ey=0.28, nx=1, ny=0.28,
           pts=[(1600, 301), (1600, 450), (675, 450)], straight=True)
    d.edge(profile, de, dashed=True, ex=1, ey=0.72, nx=1, ny=0.72,
           pts=[(1630, 349), (1630, 579), (1020, 579)], straight=True)

    d.P("p1")
    lb2 = d.box(
        490, 740, 250, 100,
        title="LB2",
        icons=[("tag:LB2:#18A9CC", 68, 36)],
        fill=TINT_C, stroke="#18A9CC", sw=3,
    )
    repair = d.box(
        90, 740, 250, 100,
        title="Repair",
        icons=[("tag:Repair:#18A9CC", 92, 36)],
        fill=TINT_C, stroke="#18A9CC", sw=3,
    )
    d.text(360, 742, 110, 36, "candidate", align="center")
    d.edge(gwo, lb2, ex=0.5, ey=1, nx=0.5, ny=0,
           pts=[(515, 650), (615, 650)], straight=True)
    d.edge(de, lb2, ex=0.5, ey=1, nx=0.82, ny=0,
           pts=[(890, 690), (695, 690)], straight=True)
    d.edge(lb2, repair, ex=0, ey=0.5, nx=1, ny=0.5, straight=True)
    d.text(85, 650, 150, 36, "if fitness ≥", align="center")
    d.edge(repair, population, ex=0, ey=0.5, nx=0, ny=0.5,
           pts=[(60, 790), (60, 115)], straight=True)
    # Compact geometry only: retain font sizes, content and directed endpoints.
    for cell in d.cells:
        if cell['kind'] == 'v':
            x, y, w, h = cell['geo']
            cell['geo'] = (x * .9, y * .78, w * .9, h * .78)
        elif cell.get('pts'):
            cell['pts'] = [(x * .9, y * .78) for x, y in cell['pts']]
    d.save(out)


if __name__ == "__main__":
    if "OUT" in globals():
        generate(OUT)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("output", nargs="?", default="figuras/arquitectura-hibrida")
        generate(parser.parse_args().output)
