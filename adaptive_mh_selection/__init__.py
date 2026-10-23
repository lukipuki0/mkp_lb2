"""Adaptive selection over the native hybrid_mkp iteration pipeline.

Use RunConfig and run_pipeline sequentially. The result dictionary contains the
unaltered PipelineResult, selection_records and a strict-JSON summary.
"""
from .config import RunConfig
from .orchestrator import run_pipeline

__all__ = ['RunConfig', 'run_pipeline']
