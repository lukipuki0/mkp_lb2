"""
Metaheuristicas poblacionales adaptadas al dominio continuo.
"""

from continuous_benchmark.mh.pso import PSOParams, ejecutar_epoch as pso_epoch  # noqa: F401
from continuous_benchmark.mh.ga  import GAParams,  ejecutar_epoch as ga_epoch   # noqa: F401
from continuous_benchmark.mh.gwo import GWOParams, ejecutar_epoch as gwo_epoch  # noqa: F401
from continuous_benchmark.mh.woa import WOAParams, ejecutar_epoch as woa_epoch  # noqa: F401
from continuous_benchmark.mh.eho import EHOParams, ejecutar_epoch as eho_epoch  # noqa: F401
from continuous_benchmark.mh.abc import ABCParams, ejecutar_epoch as abc_epoch  # noqa: F401
from continuous_benchmark.mh.aco import ACOParams, ejecutar_epoch as aco_epoch  # noqa: F401
from continuous_benchmark.mh.ils import ILSParams, ejecutar_epoch as ils_epoch  # noqa: F401
from continuous_benchmark.mh.sa  import SAParams,  ejecutar_epoch as sa_epoch   # noqa: F401

