"""
Module configuration parser for MARS report modules.

Loads and validates mars_module.yaml files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ArgumentConfig:
    """Configuration for a single module argument."""

    name: str
    """Argument name (descriptive identifier)"""

    flag: str | None
    """Command-line flag (e.g., '--out') or None for positional argument"""

    type: str
    """Argument type: 'Path', 'str', 'int', 'bool'"""

    help: str
    """Help text for this argument"""

    required: bool
    """Whether this argument is required"""

    default: Any = None
    """Default value if not provided"""

    set: bool = False
    """For optional flags: whether the flag should be set"""

    value: Any = None
    """Pre-configured value for this argument"""

    choices: list[str] | None = None
    """Valid choices for this argument (if applicable)"""

    @classmethod
    def from_dict(cls, data: dict) -> ArgumentConfig:
        """Create ArgumentConfig from YAML dict."""
        return cls(
            name=data["name"],
            flag=data.get("flag"),
            type=data["type"],
            help=data.get("help", ""),
            required=data.get("required", False),
            default=data.get("default"),
            set=data.get("set", False),
            value=data.get("value"),
            choices=data.get("choices"),
        )

    def is_positional(self) -> bool:
        """Check if this is a positional argument (no flag)."""
        return self.flag is None

    def is_boolean(self) -> bool:
        """Check if this is a boolean flag."""
        return self.type == "bool"

    def should_include(self) -> bool:
        """Check if this argument should be included when building command."""
        if self.is_boolean():
            # Boolean flags only included if set=True
            return self.set
        if self.required:
            # Required arguments always included
            return True
        # Optional non-boolean arguments included if set=True
        return self.set


@dataclass
class ModuleConfig:
    """Configuration for a single report module."""

    # Module metadata
    name: str
    """Display name for administration"""

    report_folder_name: str
    """Name of folder created in reports directory"""

    version: str
    """Module version (informational)"""

    description: str
    """Module description (informational)"""

    dependencies: list[str]
    """Required dependencies not included with MARS"""

    readme: str
    """Optional readme filename (e.g., 'README.md')"""

    scan_type: list[str]
    """When to run: ['exemplar'], ['candidate'], ['free'], or combinations"""

    target: str
    """What to scan: 'root' for scan root, or catalog name (e.g., 'Firefox Cache')"""

    entry: str
    """Entry point function name (e.g., 'ff_cache_parser')"""

    active: bool
    """Whether this module is active"""

    # Module location
    module_path: Path
    """Path to module directory"""

    yaml_path: Path
    """Path to mars_module.yaml file"""

    # Arguments configuration
    arguments: dict[str, list[ArgumentConfig]] = field(default_factory=dict)
    """Arguments grouped by entry function name"""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ModuleConfig:
        """Load and validate module configuration from YAML file.

        Args:
            yaml_path: Path to mars_module.yaml file

        Returns:
            ModuleConfig instance

        Raises:
            ValueError: If YAML is invalid or missing required fields
            FileNotFoundError: If YAML file doesn't exist
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Module YAML not found: {yaml_path}")

        with yaml_path.open() as f:
            data = yaml.safe_load(f)

        # Validate module_info section
        if "module_info" not in data:
            raise ValueError(f"Missing 'module_info' section in {yaml_path}")

        info = data["module_info"]
        required_fields = [
            "name",
            "report_folder_name",
            "version",
            "description",
            "scan_type",
            "target",
            "entry",
            "active",
        ]

        for field_name in required_fields:
            if field_name not in info:
                raise ValueError(f"Missing required field '{field_name}' in module_info: {yaml_path}")

        # Validate scan_type is a list
        if not isinstance(info["scan_type"], list):
            raise ValueError(f"scan_type must be a list, got {type(info['scan_type'])}: {yaml_path}")

        # Validate scan_type values
        valid_scan_types = {"exemplar", "candidate", "free"}
        for st in info["scan_type"]:
            if st not in valid_scan_types:
                raise ValueError(f"Invalid scan_type '{st}', must be one of {valid_scan_types}: {yaml_path}")

        # Parse arguments section
        arguments_dict = {}
        if "arguments" in data:
            for func_name, args_list in data["arguments"].items():
                if not isinstance(args_list, list):
                    raise ValueError(f"Arguments for '{func_name}' must be a list: {yaml_path}")

                arguments_dict[func_name] = [ArgumentConfig.from_dict(arg) for arg in args_list]

        return cls(
            name=info["name"],
            report_folder_name=info["report_folder_name"],
            version=info["version"],
            description=info["description"],
            dependencies=info.get("dependencies", []),
            readme=info.get("readme", ""),
            scan_type=info["scan_type"],
            target=info["target"],
            entry=info["entry"],
            active=info["active"],
            module_path=yaml_path.parent,
            yaml_path=yaml_path,
            arguments=arguments_dict,
        )

    def matches_scan_type(self, scan_type: str) -> bool:
        """Check if this module should run for the given scan type.

        Args:
            scan_type: 'exemplar', 'candidate', or 'free'

        Returns:
            True if module should run for this scan type
        """
        return scan_type in self.scan_type

    def get_arguments(self, func_name: str | None = None) -> list[ArgumentConfig]:
        """Get arguments for a specific function or the entry function.

        Args:
            func_name: Function name to get arguments for (defaults to entry)

        Returns:
            List of ArgumentConfig objects
        """
        if func_name is None:
            func_name = self.entry

        return self.arguments.get(func_name, [])

    def validate(self) -> list[str]:
        """Validate module configuration and return list of errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check if module directory exists
        if not self.module_path.exists():
            errors.append(f"Module directory does not exist: {self.module_path}")

        # Check if entry module can be imported
        try:
            entry_file = self.module_path / f"{self.entry}.py"
            if not entry_file.exists():
                errors.append(f"Entry file not found: {entry_file}")
        except Exception as e:
            errors.append(f"Error checking entry file: {e}")

        # Check required arguments have no default value set
        for func_name, args in self.arguments.items():
            for arg in args:
                if arg.required and arg.value is None:
                    errors.append(f"Required argument '{arg.name}' in '{func_name}' has no value")

        return errors

    def __str__(self) -> str:
        """String representation for debugging."""
        active_str = "ACTIVE" if self.active else "INACTIVE"
        return f"Module({self.name} v{self.version}) [{active_str}] scan_type={self.scan_type} target={self.target}"
