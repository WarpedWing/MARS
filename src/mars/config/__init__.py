"""MARS Configuration Module.

This module provides centralized configuration management for MARS,
replacing scattered hardcoded values throughout the codebase.

Usage:
    >>> from mars.config import MARSConfig, ConfigLoader, ProjectPaths
    >>>
    >>> # Load with defaults
    >>> config = MARSConfig()
    >>>
    >>> # Load with priority: CLI > Project > Defaults
    >>> config = ConfigLoader.load(
    ...     cli_args={"matching": {"min_confidence": 0.8}},
    ...     project_dir=Path.cwd(),
    ... )
    >>>
    >>> # Create auto-named project paths
    >>> paths = ProjectPaths.create(
    ...     base_dir=Path("/output"),
    ...     output_config=config.output,
    ...     case_name="MyCase"
    ... )
    >>> paths.create_all()
"""

from mars.config.loader import ConfigLoader
from mars.config.paths import ProjectPaths
from mars.config.schema import (
    MARSConfig,
    MatchingConfig,
    OutputConfig,
    ScannerConfig,
    SemanticAnchorWeights,
    UIConfig,
)

__all__ = [
    # Main configuration class
    "MARSConfig",
    # Configuration subsections
    "MatchingConfig",
    "SemanticAnchorWeights",
    "ScannerConfig",
    "OutputConfig",
    "UIConfig",
    # Utilities
    "ConfigLoader",
    "ProjectPaths",
]

__version__ = "1.0.0"
