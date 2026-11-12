"""Structural checks for the redistributed hybrid architecture.

python figuras/arquitectura-hibrida/layout.py
Content hashes exclude style and geometry while retaining every endpoint.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EDGES = {
    "population-alpha": ("Shared population", "Alpha"),
    "population-beta": ("Shared population", "Beta"),
    "population-delta": ("Shared population", "Delta"),
    "population-monitor": ("Shared population", "Monitor"),
    "population-de": ("Shared population", "DE"),
    "alpha-gwo": ("Alpha", "GWO"),
    "beta-gwo": ("Beta", "GWO"),
    "delta-gwo": ("Delta", "GWO"),
    "monitor-profile": ("Monitor", "Profile"),
    "profile-gwo": ("Profile", "GWO"),
    "profile-de": ("Profile", "DE"),
    "gwo-lb2": ("GWO", "LB2"),
    "de-lb2": ("DE", "LB2"),
    "lb2-repair": ("LB2", "Repair"),
    "repair-population": ("Repair", "Shared population"),
}
LABELS = {
    "copied, frozen", "guided", "individual i", "candidate",
    "if fitness ≥", "curve", "profile",
}


def styles(cell):
    return dict(item.split("=", 1) for item in cell.get("style", "").split(";") if "=" in item)


def plain(value):
    text = (value or "").replace("<br>", " ")
    while "<" in text and ">" in text:
        text = text[:text.find("<")] + " " + text[text.find(">") + 1:]
    return " ".join(text.replace("&gt;", ">").replace("&lt;", "<").split())


def title_of(cell):
    text = plain(cell.get("value"))
    for marker in (" continuous ", " parameter ", " a, ", " F, "):
        text = text.split(marker)[0]
    return text


def rect(cell):
    geometry = cell.find("mxGeometry")
    return tuple(float(geometry.get(key, "0")) for key in ("x", "y", "width", "height"))


def digest(cells):
    data = [{key: value for key, value in cell.attrib.items() if key != "style"} for cell in cells]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def hits_rect(start, end, box):
    x, y, width, height = box
    low, high = 0.0, 1.0
    for begin, finish, lower, upper in (
        (start[0], end[0], x, x + width),
        (start[1], end[1], y, y + height),
    ):
        delta = finish - begin
        if abs(delta) < 1e-8:
            if not lower < begin < upper:
                return False
        else:
            first, second = sorted(((lower - begin) / delta, (upper - begin) / delta))
            low, high = max(low, first), min(high, second)
    return low < high - 1e-8


def cross(left, right):
    return left[0] * right[1] - left[1] * right[0]


def subtract(left, right):
    return left[0] - right[0], left[1] - right[1]


def segment_relation(start, end, other_start, other_end):
    direction, other, offset = subtract(end, start), subtract(other_end, other_start), subtract(other_start, start)
    determinant = cross(direction, other)
    if abs(determinant) < 1e-8:
        if abs(cross(offset, direction)) < 1e-8:
            axis = 0 if abs(direction[0]) > abs(direction[1]) else 1
            if max(min(start[axis], end[axis]), min(other_start[axis], other_end[axis])) < min(max(start[axis], end[axis]), max(other_start[axis], other_end[axis])) - 1e-8:
                return "overlap"
    else:
        first, second = cross(offset, other) / determinant, cross(offset, direction) / determinant
        if 1e-8 < first < 1 - 1e-8 and 1e-8 < second < 1 - 1e-8:
            return "crossing"
    return None


def content_digest(path):
    cells = ET.parse(path).find("diagram").findall(".//mxCell")
    data = []
    for cell in cells:
        if cell.get("id", "").startswith("anchor"):
            continue
        item = {key: value for key, value in cell.attrib.items() if key not in {"style", "id"}}
        data.append(item)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def check():
    assert hits_rect((0, 0), (10, 10), (4, 4, 2, 2))
    assert not hits_rect((0, 0), (10, 10), (0, 5, 2, 2))
    assert segment_relation((0, 0), (10, 10), (0, 10), (10, 0)) == "crossing"
    assert segment_relation((0, 0), (10, 10), (5, 5), (15, 15)) == "overlap"
    pages = {}
    for path in sorted(HERE.glob("*.drawio")):
        for page in ET.parse(path).findall("diagram"):
            cells = page.findall(".//mxCell")
            pages[(path.name, page.get("id"))] = digest(cells)
            by_id = {cell.get("id"): cell for cell in cells}
            boxes = [cell for cell in cells if cell.get("vertex") and "strokeWidth=" in cell.get("style", "") and "fontColor=#FFFFFF" not in cell.get("style", "")]
            labels = [cell for cell in cells if cell.get("style", "").startswith("text;")]
            badges = [cell for cell in cells if "fontColor=#FFFFFF" in cell.get("style", "")]
            assert {title_of(cell) for cell in boxes} == {
                "Shared population", "Alpha", "Beta", "Delta", "GWO", "DE", "LB2", "Repair", "Monitor", "Profile"
            }
            assert {plain(cell.get("value")) for cell in labels} == LABELS
            assert all("Source Sans 3" in cell.get("style", "") for cell in boxes + labels + badges)
            assert all("fontSource=" in cell.get("style", "") for cell in boxes + labels + badges)
            named = {title_of(cell): cell for cell in boxes}
            assert max(rect(c)[1] + rect(c)[3] for c in boxes) < 700, 'Layout must be compact'
            return_edge = next(c for c in cells if c.get('edge') and title_of(by_id[c.get('source')]) == 'Repair')
            assert styles(return_edge)['exitX'] == '0', 'Repair return must leave its left side'
            edges = []
            for edge in [cell for cell in cells if cell.get("edge")]:
                source, target = title_of(by_id[edge.get("source")]), title_of(by_id[edge.get("target")])
                edges.append((source, target))
                points = []
                for endpoint, prefix in (("source", "exit"), ("target", "entry")):
                    x, y, width, height = rect(by_id[edge.get(endpoint)])
                    style = styles(edge)
                    points.append((x + width * float(style[prefix + "X"]), y + height * float(style[prefix + "Y"])))
                points[1:1] = [(float(point.get("x")), float(point.get("y"))) for point in edge.findall(".//Array/mxPoint")]
                for start, end in zip(points, points[1:]):
                    assert abs(start[0] - end[0]) < 1.0 or abs(start[1] - end[1]) < 1.0, (edge.get("id"), "non-orthogonal")
                    for node in boxes + labels:
                        if node.get("id") in (edge.get("source"), edge.get("target")):
                            continue
                        assert not hits_rect(start, end, rect(node)), (edge.get("id"), "intersects", plain(node.get("value")))
            assert set(edges) == set(EDGES.values())
            assert rect(named["LB2"])[1] == rect(named["Repair"])[1]
            assert rect(named["Repair"])[0] + rect(named["Repair"])[2] < rect(named["LB2"])[0], 'Repair must be left of LB2'
            assert rect(named["Monitor"])[0] > rect(named["Shared population"])[0] + rect(named["Shared population"])[2]
            assert rect(named["Profile"])[0] == rect(named["Monitor"])[0]
            print(path.name, page.get("id"), "PASS")
    main = HERE / "GWODE-ArquitecturaHibrida-v3.drawio"
    full = HERE / "page-full.drawio"
    assert content_digest(main) == content_digest(full), "main/full semantic content diverged"
    for page_id in ("p1", "p2"):
        page = ET.parse(HERE / f"page-{page_id}.drawio").find("diagram")
        assert page is not None, page_id
        assert {cell.get("edge") for cell in page.findall(".//mxCell") if cell.get("edge")} == {"1"}
    print("PASS: semantic graph, orthogonal clearance, aligned rows, Source Sans")


if __name__ == "__main__":
    check()
