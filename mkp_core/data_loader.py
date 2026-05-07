"""
data_loader.py
──────────────
Responsabilidad única: descargar y parsear instancias MKP desde una URL
(formato OR-Library de Beasley).

Formato esperado:
  <num_instancias>
  Para cada instancia:
    <n>  <m>  <valor_optimo>
    <p_1> ... <p_n>          ← ganancias
    <r_i_1> ... <r_i_n>     ← pesos de restricción i  (m filas)
    <b_1> ... <b_m>          ← capacidades
"""

from __future__ import annotations

import numpy as np
import requests


# ── Tipos ─────────────────────────────────────────────────────────────────────
Instance = dict  # {"n", "m", "valor_optimo", "p", "r", "b"}


# ── Descarga ──────────────────────────────────────────────────────────────────

def descargar_texto(url: str) -> str:
    """Descarga el contenido de *url* y lo devuelve como cadena de texto.

    Raises:
        requests.exceptions.RequestException: si la petición falla.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


# ── Parseo ────────────────────────────────────────────────────────────────────

def parsear_instancias(cadena: str) -> list[Instance]:
    """Parsea el texto crudo de OR-Library y devuelve una lista de instancias.

    Cada instancia es un dict con claves:
        n            – número de ítems
        m            – número de restricciones (mochilas)
        valor_optimo – valor óptimo conocido
        p            – array (n,)   de ganancias
        r            – array (m, n) de pesos por restricción
        b            – array (m,)   de capacidades
    """
    lineas = cadena.strip().split("\n")
    numeros: list[list[float]] = [
        list(map(float, linea.split()))
        for linea in lineas
        if linea.strip()          # ignorar líneas vacías
    ]

    ptr = 0                        # puntero a la línea actual
    n_instancias = int(numeros[ptr][0])
    ptr += 1

    instancias: list[Instance] = []

    for _ in range(n_instancias):
        n, m, valor_optimo = numeros[ptr]
        n, m = int(n), int(m)
        ptr += 1

        # Ganancias
        p: list[float] = []
        while len(p) < n:
            p.extend(numeros[ptr])
            ptr += 1

        # Pesos de cada restricción
        r_filas: list[list[float]] = []
        for _ in range(m):
            fila: list[float] = []
            while len(fila) < n:
                fila.extend(numeros[ptr])
                ptr += 1
            r_filas.append(fila)

        # Capacidades
        b: list[float] = []
        while len(b) < m:
            b.extend(numeros[ptr])
            ptr += 1

        instancias.append(
            {
                "n": n,
                "m": m,
                "valor_optimo": valor_optimo,
                "p": np.array(p),
                "r": np.array(r_filas),
                "b": np.array(b),
            }
        )

    return instancias


# ── API pública ───────────────────────────────────────────────────────────────

def cargar_instancias(url: str) -> list[Instance]:
    """Descarga y parsea todas las instancias desde *url*."""
    texto = descargar_texto(url)
    instancias = parsear_instancias(texto)
    print(f"[data_loader] {len(instancias)} instancias cargadas desde {url}")
    return instancias


def seleccionar_instancia(instancias: list[Instance], indice: int) -> Instance:
    """Devuelve la instancia en *indice* con un mensaje de resumen."""
    inst = instancias[indice]
    print(
        f"[data_loader] Instancia #{indice}: "
        f"n={inst['n']}, m={inst['m']}, óptimo={inst['valor_optimo']}"
    )
    return inst
