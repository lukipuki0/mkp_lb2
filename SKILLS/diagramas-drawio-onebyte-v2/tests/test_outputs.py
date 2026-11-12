"""Contratos públicos del generador, sin dependencias de renderizado."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class OutputTests(unittest.TestCase):
    def test_multipage_editable_and_fade(self):
        from dgen import D
        d = D('prueba', 'Título & prueba')
        d.text(20, 0, 300, 40, 'Encabezado')
        d.P('base')
        base = d.text(20, 40, 300, 40, 'Base')
        d.P('p1', 'Parte 1')
        a = d.box(20, 100, 240, 80, title='A & B', sub='Detalle')
        d.P('p2', 'Parte 2')
        b = d.box(320, 100, 240, 80, title='Destino')
        d.edge(a, b, label='Flujo')
        with tempfile.TemporaryDirectory() as tmp:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = d.save(tmp, windows={'p1': 'with-base'})
            self.assertIsNone(result)
            self.assertEqual((stdout.getvalue(), stderr.getvalue()), ('', ''))
            self.assertEqual({p.name for p in Path(tmp).iterdir()},
                             {'prueba.drawio', 'page-full.drawio', 'page-p1.drawio', 'page-p2.drawio', 'meta.json'})
            pages = ET.parse(Path(tmp) / 'prueba.drawio').getroot().findall('diagram')
            self.assertEqual([p.get('name') for p in pages], ['Título & prueba · completa', 'Parte 1', 'Parte 2'])
            for page in pages:
                cells = {c.get('id'): c for c in page.findall('.//mxCell')}
                self.assertIn(base, cells)
                self.assertEqual(cells[a].get('value'), '<b style="font-size:24px;color:#12203F">A & B</b><br><span style="font-size:21px;color:#2B3F66">Detalle</span>')
                self.assertEqual(len(page.findall('.//mxCell[@edge="1"]')), 1)
            p1 = {c.get('id'): c for c in pages[1].findall('.//mxCell')}
            self.assertNotIn('opacity=16', p1[a].get('style'))
            self.assertIn('opacity=16;textOpacity=16;', p1[b].get('style'))
            meta = json.loads((Path(tmp) / 'meta.json').read_text())
            self.assertEqual(meta['parts'], [['p1', 'Parte 1'], ['p2', 'Parte 2']])
            self.assertEqual(meta['windows'], {'p1': [20, 40, 320, 180], 'p2': [320, 100, 560, 180]})
            for key in ('full', 'p1', 'p2'):
                self.assertEqual(len(ET.parse(Path(tmp) / f'page-{key}.drawio').getroot()), 1)

    def test_single_part_stays_single_page(self):
        from dgen import D
        d = D('simple', 'Simple')
        d.P('p1', 'Parte única')
        d.box(20, 20, 200, 80, title='Única')
        with tempfile.TemporaryDirectory() as tmp:
            d.save(tmp)
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, {'simple.drawio', 'page-full.drawio', 'meta.json'})
            self.assertEqual(len(ET.parse(Path(tmp) / 'simple.drawio').getroot()), 1)


    def test_icons_are_native_drawio(self):
        from icons import S, style
        for alias, st in S.items():
            self.assertTrue('img/lib/azure2/' in st or 'shape=mxgraph.' in st, alias)
            self.assertNotIn('http', st)
        with self.assertRaises(KeyError):
            style('inventado')


if __name__ == '__main__':
    unittest.main()
