"""
MARS Report Modules System.

Provides a plugin architecture for post-processing report modules.
"""

from mars.report_modules.argument_builder import ArgumentBuilder
from mars.report_modules.module_config import ArgumentConfig, ModuleConfig
from mars.report_modules.module_runner import ModuleResult, ModuleRunner
from mars.report_modules.progress_interface import (
    ModuleProgress,
    get_progress,
)
from mars.report_modules.report_module_manager import ReportModuleManager
from mars.report_modules.target_resolver import TargetResolver

__all__ = [
    "ArgumentBuilder",
    "ArgumentConfig",
    "ModuleConfig",
    "ModuleProgress",
    "ModuleResult",
    "ModuleRunner",
    "ReportModuleManager",
    "TargetResolver",
    "get_progress",
]
