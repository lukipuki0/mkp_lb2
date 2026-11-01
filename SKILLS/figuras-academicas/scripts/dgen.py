"""Generador local de XML draw.io editable, con partes opcionales."""
import base64, html, json
from pathlib import Path
IC = Path.cwd() / 'ic'
FONT = 'fontFamily=Source Sans 3;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DSource%2BSans%2B3;'
INK, MUTED, BLUE, CYAN, NAVY, GREY = '#12203F', '#2B3F66', '#1E4FD0', '#18C0E4', '#0B2252', '#8A9BB8'
TINT_B, TINT_C, NOTE = '#E3ECFB', '#E3F6FB', '#EEF2F8'
AMBER, AMBER_T, RED, RED_T, GREEN, GREEN_T = '#B26A00', '#FFF3DC', '#B42318', '#FDECEA', '#1F7A45', '#E6F4EC'
FADE = 'opacity=16;textOpacity=16;'

class D:
    def __init__(self, name, title):
        self.name, self.title = name, title
        self.cells, self.n, self.part, self.parts = [], 1, 'hdr', []
    def P(self, key, label=None):
        self.part = key
        if label and key not in [k for k, _ in self.parts]: self.parts.append((key, label))
    def nid(self, p='c'):
        self.n += 1; return f'{p}{self.n}'
    def vertex(self, x, y, w, h, value, style):
        cid = self.nid(); self.cells.append(dict(part=self.part, kind='v', id=cid, value=value, style=style, geo=(x, y, w, h))); return cid
    def img(self, x, y, w, h, name):
        if name.startswith('tag:'):
            _, txt, bg = name.split(':', 2)
            return self.vertex(x, y, w, h, txt, f'rounded=1;arcSize=6;absoluteArcSize=1;html=1;fillColor={bg};strokeColor=none;fontColor=#FFFFFF;fontStyle=1;fontSize=17;{FONT}align=center;verticalAlign=middle;')
        data = base64.b64encode((IC / f'{name}.svg').read_bytes()).decode()
        return self.vertex(x, y, w, h, '', f'shape=image;imageAspect=1;aspect=fixed;verticalLabelPosition=bottom;image=data:image/svg+xml,{data};')
    @staticmethod
    def iw(icons, gap=10): return sum(w for _, w, _ in icons) + gap * (len(icons) - 1) if icons else 0
    def icons_row(self, x, cy, icons, gap=10):
        for name, w, h in icons: self.img(x, cy - h / 2, w, h, name); x += w + gap
    def box(self, x, y, w, h, title=None, sub=None, icons=(), fill='#FFFFFF', stroke=INK, sw=2, ts=24, ss=21,
            align='left', radius=10, dashed=False, icons_top=False, tcolor=INK, valign='middle'):
        pad, icw = 18, self.iw(icons)
        lab = '<br>'.join(([f'<b style="font-size:{ts}px;color:{tcolor}">{title}</b>'] if title else []) + ([f'<span style="font-size:{ss}px;color:{MUTED}">{sub}</span>'] if sub else []))
        sl = pad + (icw + 16 if icw and not icons_top else 0)
        st = (f'rounded=1;arcSize={radius};absoluteArcSize=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth={sw};'
              f'{FONT}fontColor={INK};fontSize={ss};align={align};verticalAlign={"bottom" if icons_top else valign};spacingTop={14 if valign == "top" else 0};'
              f'spacingLeft={sl};spacingRight={pad};spacingBottom={14 if icons_top else 0};' + ('dashed=1;dashPattern=8 6;' if dashed else ''))
        cid = self.vertex(x, y, w, h, lab, st)
        if icons_top: self.icons_row(x + (w - icw) / 2, y + 22 + max(hh for _, _, hh in icons) / 2, icons)
        else: self.icons_row(x + pad, y + h / 2, icons)
        return cid
    def container(self, x, y, w, h, title, icons=(), stroke=BLUE, fill='#FFFFFF', ts=24, dashed=False):
        pad, icw = 20, self.iw(icons)
        st = (f'rounded=1;arcSize=18;absoluteArcSize=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=3;{FONT}fontColor={NAVY};fontSize={ts};fontStyle=1;'
              f'align=left;verticalAlign=top;spacingTop=12;spacingLeft={pad + (icw + 14 if icw else 0)};' + ('dashed=1;dashPattern=10 6;' if dashed else ''))
        cid = self.vertex(x, y, w, h, title, st); self.icons_row(x + pad, y + 30, icons); return cid
    def badge(self, x, y, w, h, text, icons=(), fill='#FFFFFF', stroke='#B9C6DC', ss=20):
        return self.box(x, y, w, h, sub=text, icons=icons, fill=fill, stroke=stroke, sw=1, ss=ss, radius=min(29, h // 2))
    def num(self, x, y, n, fill=BLUE, d=44):
        return self.vertex(x, y, d, d, str(n), f'ellipse;html=1;fillColor={fill};strokeColor=none;fontColor=#FFFFFF;fontStyle=1;fontSize=22;{FONT}align=center;verticalAlign=middle;')
    def text(self, x, y, w, h, value, size=21, color=MUTED, align='left', bold=False):
        return self.vertex(x, y, w, h, value, f'text;html=1;whiteSpace=wrap;{FONT}fontSize={size};fontColor={color};align={align};verticalAlign=middle;fontStyle={1 if bold else 0};')
    def edge(self, src, tgt, color=BLUE, sw=3, both=False, arrow=True, label='', ex=None, ey=None, nx=None, ny=None, dashed=False, straight=False, pts=None):
        st = (f'edgeStyle={"none" if straight else "orthogonalEdgeStyle"};rounded=1;html=1;strokeColor={color};strokeWidth={sw};endArrow={"block" if arrow else "none"};endFill=1;'
              f'startArrow={"block" if both else "none"};startFill=1;{FONT}fontSize=20;fontColor={MUTED};labelBackgroundColor=#F4F7FB;' + ('dashed=1;dashPattern=8 6;' if dashed else ''))
        for k, v in (('exitX', ex), ('exitY', ey), ('entryX', nx), ('entryY', ny)):
            if v is not None: st += f'{k}={v};'
        self.cells.append(dict(part=self.part, kind='e', id=self.nid('e'), value=label, style=st, src=src, tgt=tgt, pts=pts))
    def extent(self):
        g = [c['geo'] for c in self.cells if c['kind'] == 'v']
        return max(a[0] + a[2] for a in g) + 20, max(a[1] + a[3] for a in g) + 20
    def bbox(self, parts):
        g = [c['geo'] for c in self.cells if c['kind'] == 'v' and c['part'] in parts]
        return (min(a[0] for a in g), min(a[1] for a in g), max(a[0] + a[2] for a in g), max(a[1] + a[3] for a in g))
    def _page(self, name, pid, mode):
        vis, out = set(), []
        X1, Y1 = self.extent()
        for i, (ax, ay) in enumerate(((0, 0), (X1, Y1))):  # anclas invisibles: fijan el origen del render
            out.append(f'<mxCell id="anchor{i}" value="" style="rounded=0;fillColor=none;strokeColor=none;" vertex="1" parent="1"><mxGeometry x="{ax}" y="{ay}" width="1" height="1" as="geometry"/></mxCell>')
        for c in self.cells:
            if c['kind'] != 'v' or not mode.get(c['part']): continue
            vis.add(c['id']); st = c['style'] + (FADE if mode[c['part']] == 'fade' else ''); x, y, w, h = c['geo']
            out.append(f'<mxCell id="{c["id"]}" value="{html.escape(c["value"], quote=True)}" style="{html.escape(st, quote=True)}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        for c in self.cells:
            if c['kind'] != 'e' or not mode.get(c['part']) or c['src'] not in vis or c['tgt'] not in vis: continue
            st = c['style'] + (FADE if mode[c['part']] == 'fade' else '')
            pts = ('<Array as="points">' + ''.join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in c['pts']) + '</Array>') if c['pts'] else ''
            out.append(f'<mxCell id="{c["id"]}" value="{html.escape(c["value"], quote=True)}" style="{html.escape(st, quote=True)}" edge="1" parent="1" source="{c["src"]}" target="{c["tgt"]}"><mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
        return (f'<diagram id="{pid}" name="{html.escape(name, quote=True)}"><mxGraphModel grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1720" pageHeight="1640" background="#F4F7FB" math="0" shadow="0">'
                '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(out) + '</root></mxGraphModel></diagram>')
    def save(self, outdir, windows=None):
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        keys = [k for k, _ in self.parts]
        pages = [(self.title + ' · completa', 'full', {k: 'show' for k in ['hdr', 'base'] + keys})]
        if len(keys) > 1:
            for k, lab in self.parts:
                m = {q: 'fade' for q in keys}; m[k] = 'show'; m['base'] = 'show'; pages.append((lab, k, m))
        head = '<mxfile host="app.diagrams.net" agent="OneByte" version="24.0.0">'
        (outdir / f'{self.name}.drawio').write_text(head + ''.join(self._page(*p) for p in pages) + '</mxfile>', encoding='utf-8')
        for p in pages: (outdir / f'page-{p[1]}.drawio').write_text(head + self._page(*p) + '</mxfile>', encoding='utf-8')
        win = {}
        for k in keys:
            v = (windows or {}).get(k)
            win[k] = self.bbox(['base', k]) if v == 'with-base' else (v or self.bbox([k]))
        (outdir / 'meta.json').write_text(json.dumps({'name': self.name, 'title': self.title, 'parts': self.parts, 'extent': self.extent(), 'windows': win}), encoding='utf-8')
