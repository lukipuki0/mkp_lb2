"""Muestra técnica stdlib: dos partes, sin representar arquitectura del informe."""
import argparse
from dgen import D, BLUE, NAVY


def generate(out):
    d = D('Muestra-Diagrama-v3', 'Muestra técnica · versión 3')
    d.text(20, 0, 1000, 40, 'Muestra técnica · versión 3', size=24, color=NAVY)
    d.P('base')
    d.text(20, 40, 1000, 40, 'Ejemplo ficticio; no es un entregable del informe')
    d.P('p1', 'Parte 1 · Origen')
    a = d.box(40, 120, 400, 100, title='Origen de muestra', sub='Dato ficticio',
              icons=[('tag:Origen:#1E4FD0', 100, 36)], stroke=BLUE)
    d.P('p2', 'Parte 2 · Destino')
    b = d.box(560, 120, 400, 100, title='Destino de muestra', sub='Dato ficticio',
              icons=[('tag:Destino:#0B2252', 110, 36)], stroke=BLUE)
    d.edge(a, b, ex=1, ey=0.5, nx=0, ny=0.5)
    d.save(out, windows={'p1': 'with-base', 'p2': 'with-base'})


if __name__ == '__main__':
    if 'OUT' in globals():
        generate(OUT)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('output', help='Directorio de salida de la muestra')
        generate(parser.parse_args().output)
