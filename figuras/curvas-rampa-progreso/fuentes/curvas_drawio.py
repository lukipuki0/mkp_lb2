"""Ramp, progress and plateau curves of the conference slides as editable draw.io figures.

Uses the shared `D` API from SKILLS/diagramas-drawio-onebyte-v2. Curves are
polyline edges between invisible anchors, so every line stays editable in draw.io.
They are illustrative patterns, not experimental results.
"""
import argparse
import math
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[3] / "SKILLS" / "diagramas-drawio-onebyte-v2" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from dgen import (AMBER, AMBER_T, BLUE, FONT, GREEN, GREEN_T, INK, MUTED,  # noqa: E402
                  NAVY, D)

HIDDEN = "rounded=0;fillColor=none;strokeColor=none;"


class Plot:
    """Map data coordinates onto a pixel rectangle of a D diagram."""

    def __init__(self, d, x, y, w, h, xr, yr):
        self.d, self.x, self.y, self.w, self.h = d, x, y, w, h
        self.xr, self.yr = xr, yr

    def px(self, u, v):
        fx = (u - self.xr[0]) / (self.xr[1] - self.xr[0])
        fy = (v - self.yr[0]) / (self.yr[1] - self.yr[0])
        return round(self.x + fx * self.w), round(self.y + self.h - fy * self.h)

    def anchor(self, u, v):
        x, y = self.px(u, v)
        return self.d.vertex(x, y, 1, 1, "", HIDDEN)

    def axes(self, xlabel, ylabel):
        d = self.d
        o = d.vertex(self.x, self.y + self.h, 1, 1, "", HIDDEN)
        xe = d.vertex(self.x + self.w + 30, self.y + self.h, 1, 1, "", HIDDEN)
        ye = d.vertex(self.x, self.y - 30, 1, 1, "", HIDDEN)
        d.edge(o, xe, color=NAVY, sw=3, straight=True)
        d.edge(o, ye, color=NAVY, sw=3, straight=True)
        d.text(self.x, self.y + self.h + 14, self.w, 34, xlabel, size=22, color=INK, align="center")
        d.vertex(self.x - 64, self.y, 40, self.h, ylabel,
                 f"text;html=1;whiteSpace=wrap;{FONT}fontSize=22;fontColor={INK};"
                 "align=center;verticalAlign=middle;horizontal=0;")

    def line(self, pts, color, sw=5, dashed=False, markers=False):
        d = self.d
        a, b = self.anchor(*pts[0]), self.anchor(*pts[-1])
        d.edge(a, b, color=color, sw=sw, arrow=False, straight=True, dashed=dashed,
               pts=[self.px(u, v) for u, v in pts[1:-1]])
        if markers:
            for u, v in pts:
                x, y = self.px(u, v)
                d.vertex(x - 8, y - 8, 16, 16, "", f"ellipse;html=1;fillColor={color};strokeColor=#FFFFFF;strokeWidth=2;")

    def span(self, u0, u1, fill):
        x0, y0 = self.px(u0, self.yr[1])
        x1, y1 = self.px(u1, self.yr[0])
        self.d.vertex(x0, y0, x1 - x0, y1 - y0, "", f"rounded=0;html=1;fillColor={fill};strokeColor=none;")

    def gap(self, u, v0, v1, color):
        a, b = self.anchor(u, v0), self.anchor(u, v1)
        self.d.edge(a, b, color=color, sw=3, both=True, straight=True)


def curva_progreso_estancamiento(out):
    d = D("curva-progreso-estancamiento-v3", "Progreso y estancamiento · versión 3")
    d.text(40, 20, 1200, 40, "La curva de convergencia: progreso y estancamiento", size=24, color=NAVY, bold=True)
    d.P("base")
    p = Plot(d, 140, 110, 1080, 520, (0, 100), (0, 100))
    p.span(0, 35, GREEN_T)
    p.span(35, 100, AMBER_T)
    p.axes("Iteración", "Mejor resultado encontrado")
    p.line([(x, 10 + 78 * (1 - math.exp(-x / 9))) for x in range(0, 101, 2)], BLUE, sw=6)
    x, y = p.px(17.5, 30)
    d.text(x - 120, y - 20, 240, 40, "progreso", size=24, color=GREEN, align="center", bold=True)
    d.text(x - 120, y + 22, 240, 34, "mejora rápido", size=21, color=GREEN, align="center")
    x, y = p.px(67, 52)
    d.text(x - 160, y - 20, 320, 40, "estancamiento", size=24, color=AMBER, align="center", bold=True)
    d.text(x - 160, y + 22, 320, 34, "deja de mejorar", size=21, color=AMBER, align="center")
    d.save(out / "curva-progreso-estancamiento")


def patrones_rampa_meseta(out):
    d = D("patrones-rampa-meseta-v3", "Patrones de comparación · versión 3")
    d.text(40, 20, 1500, 40, "Dos patrones de comparación para la ventana", size=24, color=NAVY, bold=True)
    d.P("base")
    panels = [
        (140, [(k, 1 + 0.9 * k) for k in range(10)], GREEN, GREEN_T,
         "Progreso (rampa)", "el mejor resultado sigue mejorando"),
        (900, [(k, 6.0) for k in range(10)], AMBER, AMBER_T,
         "Estancamiento (meseta)", "el mejor resultado se mantiene igual"),
    ]
    for x0, pts, color, tint, title, sub in panels:
        d.box(x0, 80, 640, 90, title=title, sub=sub, fill=tint, stroke=color)
        p = Plot(d, x0, 230, 640, 400, (-0.5, 10), (0, 10.5))
        p.axes("Posición en la ventana", "Mejor resultado")
        p.line(pts, color, markers=True)
    d.save(out / "patrones-rampa-meseta")


def signo_delta(out):
    d = D("signo-delta-rampa-meseta-v3", "Signo de Δ · versión 3")
    d.text(40, 20, 1300, 40, "El signo de Δ = D1 − D2 indica a qué patrón se parece la ventana",
           size=24, color=NAVY, bold=True)
    d.P("base")
    p = Plot(d, 140, 110, 860, 620, (0.5, 10), (0, 10))
    p.axes("Posición en la ventana (k)", "Mejor resultado")
    window = [2, 2.7, 3.1, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3]
    p.line([(1, 2), (10, 9.2)], GREEN, dashed=True)
    p.line([(1, 2), (10, 2)], AMBER, dashed=True)
    p.line(list(zip(range(1, 11), window)), BLUE, markers=True)

    for v, label, color in ((9.2, "R  rampa", GREEN), (3.3, "X  ventana", BLUE), (2.0, "C  meseta", AMBER)):
        x, y = p.px(10, v)
        dy = -36 if v == 3.3 else (4 if v == 2.0 else -18)
        d.text(x + 24, y + dy, 200, 36, label, size=24, color=color, bold=True)

    p.gap(6.5, 3.45, 2 + 0.8 * 5.5 - 0.15, GREEN)
    x, y = p.px(6.5, 5.0)
    d.text(x + 14, y - 18, 180, 36, "D1 grande", size=22, color=GREEN, bold=True)
    p.gap(8.5, 2.12, 3.18, AMBER)
    x, y = p.px(8.5, 2.65)
    d.text(x - 200, y - 18, 186, 36, "D2 pequeña", size=22, color=AMBER, align="right", bold=True)

    d.box(1280, 150, 440, 110, title="Δ < 0 · progreso", sub="más cerca de la rampa", fill=GREEN_T, stroke=GREEN)
    d.box(1280, 300, 440, 110, title="Δ > 0 · estancamiento", sub="más cerca de la meseta", fill=AMBER_T, stroke=AMBER, sw=3)
    d.box(1280, 450, 440, 110, title="Δ = 0", sub="igual distancia", fill="#FFFFFF", stroke=MUTED)
    d.save(out / "signo-delta-rampa-meseta")


def generate(out):
    out = Path(out)
    curva_progreso_estancamiento(out)
    patrones_rampa_meseta(out)
    signo_delta(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default=str(Path(__file__).resolve().parent / "drawio"))
    generate(parser.parse_args().output)
