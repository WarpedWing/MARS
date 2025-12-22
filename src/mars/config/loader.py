"""Configuration loader with priority system.

Loads and merges configuration from multiple sources with priority:
    CLI Arguments > Project Config > Defaults
"""

import json
from pathlib import Path
from typing import Any

from mars.config.schema import MARSConfig
from mars.utils.debug_logger import logger


class ConfigLoader:
    """Load and merge configuration from multiple sources with priority."""

    # Project config filename
    PROJECT_CONFIG_FILE = ".marsproj"

    @classmethod
    def load(
        cls,
        cli_args: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ) -> MARSConfig:
        """Load configuration with priority merging.

        Priority order (highest to lowest):
        1. CLI arguments (cli_args parameter)
        2. Project config (.marsproj in project_dir)
        3. Built-in defaults (from schema.py)

        Args:
            cli_args: Dictionary of CLI argument overrides
            project_dir: Path to project directory containing .marsproj

        Returns:
            Merged MARSConfig instance

        Example:
            >>> # Load with defaults only
            >>> config = ConfigLoader.load()
            >>>
            >>> # Load with CLI args overriding defaults
            >>> config = ConfigLoader.load(cli_args={
            ...     "matching": {"min_confidence": 0.8}
            ... })
            >>>
            >>> # Load with project config
            >>> config = ConfigLoader.load(
            ...     cli_args={"ui": {"verbose": True}},
            ...     project_dir=Path("/my/project"),
            ... )
        """
        # Start with defaults
        config = MARSConfig()

        # Layer 1: Project config (if in project directory)
        if project_dir:
            project_file = project_dir / cls.PROJECT_CONFIG_FILE
            if project_file.exists():
                try:
                    project_config = cls._load_json(project_file)
                    config = cls._merge_config(config, project_config)
                except (json.JSONDecodeError, OSError) as e:
                    # Don't fail if project config is corrupted
                    logger.warning(f"Could not load project config: {e}")

        # Layer 3: CLI arguments (highest priority)
        if cli_args:
            config = cls._merge_config(config, cli_args)

        # Sync dependent settings
        # debug_log_to_file requires debug mode
        if config.ui.debug_log_to_file and not config.ui.debug:
            config.ui.debug = True
        # debug mode disables progress bars
        if config.ui.debug:
            config.ui.show_progress_bars = False

        return config

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Load JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Parsed JSON as dictionary

        Raises:
            json.JSONDecodeError: If JSON is invalid
            OSError: If file cannot be read
        """
        with Path.open(path) as f:
            return json.load(f)

    @staticmethod
    def _merge_config(base: MARSConfig, overrides: dict[str, Any]) -> MARSConfig:
        """Merge override values into base configuration.

        Args:
            base: Base configuration object
            overrides: Dictionary of values to override

        Returns:
            New MARSConfig with merged values
        """
        # Convert base to dict
        base_dict = base.to_dict()

        # Deep merge nested configs
        for key, value in overrides.items():
            if key in base_dict:
                if isinstance(base_dict[key], dict) and isinstance(value, dict):
                    # Merge nested dictionaries (e.g., matching, scanner, etc.)
                    base_dict[key] = {**base_dict[key], **value}
                else:
                    # Direct override for non-dict values
                    base_dict[key] = value
            else:
                # New key not in base
                base_dict[key] = value

        # Reconstruct config from merged dict
        return MARSConfig.from_dict(base_dict)

    @classmethod
    def save_project_config(cls, config: MARSConfig, project_dir: Path) -> None:
        """Save user-configurable settings to project .marsproj file.

        Only saves settings that users should modify (marked as user_configurable).
        Internal/expert settings are excluded to prevent accidental breakage.

        Args:
            config: Configuration to save
            project_dir: Directory to save .marsproj file in

        Raises:
            OSError: If directory doesn't exist or file cannot be written
        """
        project_file = project_dir / cls.PROJECT_CONFIG_FILE
        with Path.open(project_file, "w") as f:
            # Save only user-configurable fields (safe for editing)
            json.dump(config.to_user_configurable_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, config_file: Path) -> MARSConfig:
        """Load configuration from a specific JSON file.

        Args:
            config_file: Path to configuration JSON file

        Returns:
            MARSConfig loaded from file

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        config_dict = cls._load_json(config_file)
        config = MARSConfig.from_dict(config_dict)

        # Sync dependent settings: debug mode disables progress bars
        if config.ui.debug:
            config.ui.show_progress_bars = False

        return config

    @classmethod
    def save_to_file(cls, config: MARSConfig, config_file: Path) -> None:
        """Save configuration to a specific JSON file.

        Args:
            config: Configuration to save
            config_file: Path to save configuration to

        Raises:
            OSError: If file cannot be written
        """
        # Create parent directory if it doesn't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with Path.open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    @classmethod
    def has_project_config(cls, project_dir: Path) -> bool:
        """Check if project config file exists.

        Args:
            project_dir: Project directory to check

        Returns:
            True if .marsproj file exists in project directory
        """
        return (project_dir / cls.PROJECT_CONFIG_FILE).exists()
