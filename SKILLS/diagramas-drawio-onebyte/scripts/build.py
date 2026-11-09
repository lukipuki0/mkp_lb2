# build.py  —  python3 build.py ejemplo.py
import json, subprocess, sys, runpy, html as H
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
ROOT = Path.cwd().resolve(); script = Path(sys.argv[1]).resolve(); out = ROOT / 'out' / script.stem
runpy.run_path(str(script), run_name='__main__', init_globals={'OUT': str(out)})
meta = json.loads((out / 'meta.json').read_text()); keys = [k for k, _ in meta['parts']]
def render(src, dst):
    cfg = {'xml': src.read_text(encoding='utf-8'), 'toolbar': '', 'nav': False, 'lightbox': False, 'border': 0, 'resize': False}
    page = out / 'render.html'
    page.write_text('<!doctype html><meta charset="utf-8"><style>body{margin:0;background:#F4F7FB}</style>'
                    f'<div id="d" class="mxgraph" data-mxgraph="{H.escape(json.dumps(cfg), quote=True)}"></div>'
                    f'<script src="{(ROOT / "viewer-static.min.js").as_uri()}"></script>')
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1760, 'height': 1700}, device_scale_factor=2)
        pg.goto(page.as_uri()); pg.wait_for_selector('#d svg'); pg.wait_for_timeout(1500)
        pg.locator('#d svg').first.screenshot(path=str(dst)); b.close()
for p in ['full'] + (keys if len(keys) > 1 else []): render(out / f'page-{p}.drawio', out / f'r-{p}.png')
full = Image.open(out / 'r-full.png').convert('RGB'); full.save(out / 'full.png'); S = 2
for k in keys if len(keys) > 1 else []:
    x0, y0, x1, y1 = meta['windows'][k]; m = 22
    Image.open(out / f'r-{k}.png').crop((int((x0 - m) * S), int((y0 - m) * S), int((x1 + m) * S), int((y1 + m) * S))).save(out / f'part-{k}.png')
# mapa: diagrama completo con borde punteado y número por parte
P = 64; can = Image.new('RGB', (full.width + 2 * P, full.height + 2 * P), 'white'); can.paste(full, (P, P)); dr = ImageDraw.Draw(can)
try: font = ImageFont.truetype('DejaVuSans-Bold.ttf', 56)
except OSError: font = ImageFont.load_default()
for i, k in enumerate(keys if len(keys) > 1 else []):
    x0, y0, x1, y1 = [v * S + P for v in meta['windows'][k]]; x0 -= 10; y0 -= 10; x1 += 10; y1 += 10
    for a, b2 in (((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))):
        n = int(max(abs(b2[0] - a[0]), abs(b2[1] - a[1])) // 44) + 1
        for j in range(n):
            t0, t1 = j / n, min(1, j / n + 28 / max(1, 44 * n))
            dr.line([(a[0] + (b2[0] - a[0]) * t0, a[1] + (b2[1] - a[1]) * t0), (a[0] + (b2[0] - a[0]) * t1, a[1] + (b2[1] - a[1]) * t1)], fill=(30, 79, 208), width=6)
    dr.ellipse([x0 - 40, y0 - 40, x0 + 52, y0 + 52], fill=(30, 79, 208), outline='white', width=6)
    dr.text((x0 + 6, y0 + 6), str(i + 1), font=font, fill='white', anchor='mm')
can.thumbnail((2600, 2600)); can.save(out / 'map.png')
print('listo', out)
