"""Reproducible hybrid-only layout, preserving exact content and graph.

python figuras/arquitectura-hibrida/layout.py [--apply]
Export the four PNGs using the installed draw.io CLI, then inspect them.
Content hashes exclude style/geometry, but include every cell and endpoint.
"""
from pathlib import Path
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXPECTED = {'full': 'c1de54d832d1d020bae350d952485ae0a1d09e25bc873081646c76ef729aa890',
            'p1': '2b358f11c68596b395a609324152fd043f8328c6fc44a137aa0b09159daab818',
            'p2': '2b358f11c68596b395a609324152fd043f8328c6fc44a137aa0b09159daab818'}
BOXES = {
 'anchor0': (0, 0, 1, 1), 'anchor1': (1060, 680, 1, 1),
 'c2': (100, 20, 700, 40), 'c3': (100, 80, 560, 80),
 'c9': (100, 230, 160, 60), 'c11': (280, 230, 160, 60), 'c13': (460, 230, 160, 60),
 'c5': (250, 390, 230, 80), 'c7': (650, 390, 170, 80),
 'c15': (280, 550, 170, 60), 'c17': (600, 550, 230, 60),
 'c35': (850, 90, 220, 60), 'c37': (850, 230, 220, 80),
 'c30': (195, 175, 160, 32), 'c31': (270, 325, 85, 32),
 'c32': (680, 180, 130, 32), 'c33': (465, 540, 115, 32),
 'c34': (60, 615, 135, 32), 'c43': (695, 80, 80, 32),
 'c44': (975, 185, 85, 32),
}
BADGES = {'c4': 'c3', 'c6': 'c5', 'c8': 'c7', 'c10': 'c9', 'c12': 'c11',
          'c14': 'c13', 'c16': 'c15', 'c18': 'c17', 'c36': 'c35', 'c38': 'c37'}
# Distinct ports permit diagonal fan-in without oversized receiving boxes.
ROUTES = {
 'e19': ((80/560, 1), (.5, 0), []),
 'e20': ((260/560, 1), (.5, 0), []),
 'e21': ((440/560, 1), (.5, 0), []),
 'e22': ((.5, 1), (.15, 0), []),
 'e23': ((.5, 1), (.5, 0), []),
 'e24': ((.5, 1), (.85, 0), []),
 'e25': ((.95, 1), (.5, 0), []),
 'e26': ((.5, 1), (.5, 0), []),
 'e27': ((.5, 1), (.85, 0), []),
 'e28': ((1, .5), (0, .5), []),
 'e29': ((.5, 1), (0, .5), [(715, 660), (40, 660), (40, 120)]),
 'e39': ((1, .5), (0, .5), []),
 'e40': ((.5, 1), (.5, 0), []),
 'e41': ((0, .5), (1, 0), [(770, 340), (510, 340)]),
 'e42': ((.5, 1), (1, .5), [(960, 430)]),
}


def styles(cell):
    return dict(s.split('=', 1) for s in cell.get('style', '').split(';') if '=' in s)


def digest(cells):
    data = [{k: v for k, v in c.attrib.items() if k != 'style'} for c in cells]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def apply():
    for path in sorted(HERE.glob('*.drawio')):
        tree = ET.parse(path)
        for page in tree.findall('diagram'):
            cells = page.findall('.//mxCell')
            assert digest(cells) == EXPECTED[page.get('id')]
            byid = {c.get('id'): c for c in cells}
            for ident, (x, y, w, h) in BOXES.items():
                if ident in byid:
                    byid[ident].find('mxGeometry').attrib.update(
                        x=str(x), y=str(y), width=str(w), height=str(h))
            for badge, box in BADGES.items():
                g = byid[badge].find('mxGeometry')
                x, y, w, h = BOXES[box]
                g.set('x', str(x + 18))
                g.set('y', str(y + (h - float(g.get('height'))) / 2))
            for ident, (exit_, entry, points) in ROUTES.items():
                c = byid[ident]
                s = styles(c)
                s.update(edgeStyle='none', noEdgeStyle='1', rounded='0',
                         exitX=str(exit_[0]), exitY=str(exit_[1]),
                         entryX=str(entry[0]), entryY=str(entry[1]),
                         jumpStyle='gap', jumpSize='12')
                c.set('style', ';'.join(f'{k}={v}' for k, v in s.items()) + ';')
                g = c.find('mxGeometry')
                g.clear()
                g.attrib.update(relative='1', **{'as': 'geometry'})
                if points:
                    a = ET.SubElement(g, 'Array', {'as': 'points'})
                    for x, y in points:
                        ET.SubElement(a, 'mxPoint', x=str(x), y=str(y))
            page.find('mxGraphModel').attrib.update(pageWidth='1090', pageHeight='700')
        tree.write(path, encoding='utf-8', xml_declaration=True)


def rect(c):
    g = c.find('mxGeometry')
    return tuple(float(g.get(k, '0')) for k in ('x', 'y', 'width', 'height'))


def hits_rect(a, b, box):
    """Clip any segment (including diagonals) against an open rectangle."""
    x, y, w, h = box
    lo, hi = 0., 1.
    for start, end, lower, upper in [(a[0], b[0], x, x+w), (a[1], b[1], y, y+h)]:
        delta = end - start
        if abs(delta) < 1e-8:
            if not lower < start < upper:
                return False
        else:
            t1, t2 = sorted(((lower-start)/delta, (upper-start)/delta))
            lo, hi = max(lo, t1), min(hi, t2)
    return lo < hi - 1e-8


def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def subtract(a, b):
    return a[0]-b[0], a[1]-b[1]


def segment_relation(a, b, c, d):
    """Return overlap / crossing / none for arbitrary straight segments."""
    r, s, q = subtract(b, a), subtract(d, c), subtract(c, a)
    det = cross(r, s)
    if abs(det) < 1e-8:
        if abs(cross(q, r)) < 1e-8:
            axis = 0 if abs(r[0]) > abs(r[1]) else 1
            if max(min(a[axis], b[axis]), min(c[axis], d[axis])) < min(max(a[axis], b[axis]), max(c[axis], d[axis])) - 1e-8:
                return 'overlap'
    else:
        t, u = cross(q, s)/det, cross(q, r)/det
        if 1e-8 < t < 1-1e-8 and 1e-8 < u < 1-1e-8:
            return 'crossing'
    return None


def check():
    # Geometry regression checks: a diagonal through text must be rejected.
    assert hits_rect((0, 0), (10, 10), (4, 4, 2, 2))
    assert not hits_rect((0, 0), (10, 10), (0, 5, 2, 2))
    assert segment_relation((0, 0), (10, 10), (0, 10), (10, 0)) == 'crossing'
    assert segment_relation((0, 0), (10, 10), (5, 5), (15, 15)) == 'overlap'
    for path in sorted(HERE.glob('*.drawio')):
        for page in ET.parse(path).findall('diagram'):
            cells = page.findall('.//mxCell')
            assert digest(cells) == EXPECTED[page.get('id')], (path, 'content/endpoints changed')
            byid = {c.get('id'): c for c in cells}
            for ident, maximum in {'c3': 560, 'c5': 230, 'c7': 170, 'c15': 170,
                                   'c17': 230, 'c35': 220, 'c37': 220}.items():
                assert rect(byid[ident])[2] <= maximum, (ident, 'excessive box width')
            assert rect(byid['c15'])[1] == rect(byid['c17'])[1], 'LB2/Repair must share a row'
            assert min(rect(byid[c])[0] for c in ('c35', 'c37')) >= max(rect(byid[c])[0] + rect(byid[c])[2] for c in ('c5', 'c7', 'c15', 'c17')), 'Monitor/Profile must sit at right'
            assert rect(byid['anchor1'])[1] <= 680, 'excessive canvas height'
            segments = []
            vertices = [c for c in cells if c.get('vertex') and not c.get('id').startswith('anchor')]
            for i, c in enumerate(vertices):
                x, y, w, h = rect(c)
                for other in vertices[i+1:]:
                    if BADGES.get(c.get('id')) == other.get('id') or BADGES.get(other.get('id')) == c.get('id'):
                        continue
                    ox, oy, ow, oh = rect(other)
                    assert max(x, ox) >= min(x+w, ox+ow) or max(y, oy) >= min(y+h, oy+oh), (c.get('id'), other.get('id'), 'node/label overlap')
            for edge in [c for c in cells if c.get('edge')]:
                s = styles(edge)
                assert s.get('edgeStyle') == 'none', 'validation requires explicit straight segments'
                points = []
                for endpoint, prefix in [('source', 'exit'), ('target', 'entry')]:
                    x, y, w, h = rect(byid[edge.get(endpoint)])
                    points.append((x + w * float(s[prefix+'X']), y + h * float(s[prefix+'Y'])))
                points[1:1] = [(float(p.get('x')), float(p.get('y'))) for p in edge.findall('.//Array/mxPoint')]
                if edge.get('id') in ('e39', 'e41', 'e42'):
                    from math import dist
                    length = sum(dist(a, b) for a, b in zip(points, points[1:]))
                    assert length <= {'e39': 220, 'e41': 500, 'e42': 300}[edge.get('id')], (edge.get('id'), 'feedback route too long', length)
                for a, b in zip(points, points[1:]):
                    segments.append((edge, a, b))
                    for c in vertices:
                        if c.get('id') in [edge.get('source'), edge.get('target')] or BADGES.get(c.get('id')) in [edge.get('source'), edge.get('target')]:
                            continue
                        assert not hits_rect(a, b, rect(c)), (path.name, page.get('id'), edge.get('id'), 'intersects', c.get('id'))
            crossings = set()
            for i, (edge, a, b) in enumerate(segments):
                for other, c, d in segments[i+1:]:
                    if edge is other:
                        continue
                    relation = segment_relation(a, b, c, d)
                    assert relation != 'overlap', (edge.get('id'), other.get('id'), 'shared connector segment')
                    if relation == 'crossing':
                        assert styles(edge).get('jumpStyle') == styles(other).get('jumpStyle') == 'gap'
                        crossings.add(tuple(sorted((edge.get('id'), other.get('id')))))
            assert crossings == {('e25', 'e41')}, crossings
            print(path.name, page.get('id'), 'PASS: exact content/endpoints; compact boxes; no diagonal/orthogonal edge-node/text or shared-segment overlaps; one gap crossing; aligned LB2/Repair; short right-side feedback')
    main = ET.parse(HERE / 'GWODE-ArquitecturaHibrida-v3.drawio')
    for page in main.findall('diagram'):
        standalone = ET.parse(HERE / ('page-' + page.get('id') + '.drawio')).find('diagram')
        assert ET.tostring(page.find('mxGraphModel')) == ET.tostring(standalone.find('mxGraphModel'))
        expected_dim = {'full': set(), 'p1': set('c35 c36 c37 c38 c43 c44 e39 e40 e41 e42'.split()),
                        'p2': set('c9 c10 c11 c12 c13 c14 c15 c16 c17 c18 c30 c31 c32 c33 c34 e19 e20 e21 e22 e23 e24 e25 e26 e27 e28 e29'.split())}
        assert {c.get('id') for c in page.findall('.//mxCell') if styles(c).get('opacity') == '16'} == expected_dim[page.get('id')]
    print('PASS: main/standalone XML identical; full/p1/p2 active/dim context unchanged')


if __name__ == '__main__':
    if '--apply' in sys.argv:
        apply()
    check()
