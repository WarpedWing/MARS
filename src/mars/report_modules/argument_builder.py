"""
Argument builder for MARS report modules.

Converts YAML argument configuration to sys.argv format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from mars.report_modules.module_config import ArgumentConfig


class ArgumentBuilder:
    """Builds sys.argv-style command arguments from YAML configuration."""

    def build(
        self,
        args_config: list[ArgumentConfig],
        input_path: Path,
        output_dir: Path,
        **substitutions,
    ) -> list[str]:
        """Build command-line arguments from configuration.

        Args:
            args_config: List of ArgumentConfig objects
            input_path: Input path for the module (substitutes for first positional arg)
            output_dir: Output directory path (substitutes for --out flag)
            **substitutions: Additional template substitutions

        Returns:
            List of command arguments (sys.argv format, excluding program name)

        Examples:
            >>> builder = ArgumentBuilder()
            >>> args = builder.build(
            ...     args_config=[
            ...         ArgumentConfig(name="indir", flag=None, type="Path", required=True),
            ...         ArgumentConfig(name="outdir", flag="--out", type="Path", required=True),
            ...         ArgumentConfig(name="verbose", flag="--verbose", type="bool", set=True),
            ...     ],
            ...     input_path=Path("/input"),
            ...     output_dir=Path("/output")
            ... )
            >>> args
            ['/input', '--out', '/output', '--verbose']
        """
        result = []

        # Validate required paths are not None
        if input_path is None:
            raise ValueError("input_path cannot be None")
        if output_dir is None:
            raise ValueError("output_dir cannot be None")

        # Build substitution map
        subs = {
            "input_path": str(input_path),
            "output_path": str(output_dir),
            **substitutions,
        }

        # Separate positional and named arguments
        positional = [arg for arg in args_config if arg.is_positional()]
        named = [arg for arg in args_config if not arg.is_positional()]

        # Add positional arguments first
        for arg in positional:
            if not arg.should_include():
                continue

            value = self._resolve_value(arg, subs)
            if value is not None:
                result.append(str(value))

        # Add named arguments
        for arg in named:
            if not arg.should_include():
                continue

            # Boolean flags don't take values
            if arg.is_boolean():
                result.append(arg.flag)
                continue

            # Named arguments with values
            value = self._resolve_value(arg, subs)
            if value is not None:
                result.append(arg.flag)
                result.append(str(value))

        return result

    def _resolve_value(self, arg: ArgumentConfig, substitutions: dict[str, str]) -> str | None:
        """Resolve argument value using substitutions and defaults.

        Args:
            arg: Argument configuration
            substitutions: Template substitution map

        Returns:
            Resolved value string or None if no value available

        Priority:
            1. Pre-configured value in YAML (arg.value)
            2. Template substitution (if arg.name matches key)
            3. Default value (arg.default)
            4. None (if not required) or raise error (if required)
        """
        # Check for pre-configured value
        if arg.value is not None:
            return str(arg.value)

        # Check for template substitution
        if arg.name in substitutions:
            return substitutions[arg.name]

        # Check for default value
        if arg.default is not None:
            return str(arg.default)

        # If required and still no value, this is an error
        # But we'll return None and let the module's argparse handle it
        return None
