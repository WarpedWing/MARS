"""Configuration schema for MARS.

This module defines all configuration dataclasses with their default values.
All previously hardcoded constants are centralized here.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Global ignorable tables - SQLite system tables and metadata always ignored
# This is the canonical source for all ignorable table logic across the codebase
GLOBAL_IGNORABLE_TABLES = {
    # SQLite system tables
    "sqlite_sequence",  # SQLite autoincrement tracking
    "sqlite_stat1",  # SQLite query optimizer statistics
    "sqlite_stat2",  # SQLite query optimizer statistics (older versions)
    "sqlite_stat3",  # SQLite query optimizer statistics (older versions)
    "sqlite_stat4",  # SQLite query optimizer statistics
    # Extension tables
    "sqlean_define",  # SQLean extension metadata
    # CoreData framework tables (macOS/iOS apps)
    "meta",  # CoreData metadata
    "dbinfo",  # CoreData database info
    "z_primarykey",  # CoreData primary key tracking
    "z_metadata",  # CoreData metadata
    "z_modelcache",  # CoreData model cache
    # macOS housekeeping tables
    "integrityCheck",  # macOS integrity check marker (single 'inconsequential' column)
}


def get_user_configurable_fields(config_class: type, category: str | None = None) -> dict[str, Any]:
    """Extract fields marked as user_configurable from a dataclass.

    Args:
        config_class: Dataclass type to inspect
        category: Optional category filter ("basic", "advanced", etc.)

    Returns:
        Dict mapping field name to field info including type, default, and metadata

    Example:
        >>> ui_fields = get_user_configurable_fields(UIConfig, category="basic")
        >>> for name, info in ui_fields.items():
        ...     print(f"{name}: {info['metadata']['label']}")
        debug: Debug Mode
    """
    user_fields = {}
    for field_info in fields(config_class):
        metadata = field_info.metadata
        # Field must be user_configurable AND (no category filter OR category matches)
        if metadata.get("user_configurable", False) and (category is None or metadata.get("category") == category):
            user_fields[field_info.name] = {
                "type": field_info.type,
                "default": field_info.default,
                "metadata": metadata,
            }
    return user_fields


@dataclass
class MatchingConfig:
    """Configuration for database matching and classification.

    These values control how databases are matched against exemplars
    and how confidence scores are calculated.
    """

    # Core confidence thresholds
    min_confidence: float = 0.7
    """Minimum confidence for a match to be considered valid (0.0-1.0)"""

    exemplar_confidence_threshold: float = 0.85
    """Threshold to use exemplar rubrics instead of self-match (0.0-1.0)"""

    fallback_confidence_threshold: float = 0.6
    """Minimum confidence for fallback matches (0.0-1.0)"""

    catalog_confidence_threshold: float = 0.70
    """Minimum confidence for catalog matches (higher than self-match) (0.0-1.0)"""

    catalog_confidence_threshold_high_anchors: float = 0.65
    """Catalog confidence threshold when semantic anchors >= 2.0 (0.0-1.0)"""

    self_match_confidence_threshold: float = 0.30
    """Minimum confidence for self-matches (same database schema, lower threshold)"""

    # Row/column requirements
    min_total_rows: int = 5
    """Minimum total rows in lost_and_found for reconstitution attempts"""

    min_rows: int = 10
    """Minimum number of rows required for a valid match"""

    min_self_match_rows: int = 3
    """Minimum matched rows for self-match with empty schemas (relaxed threshold)"""

    min_columns: int = 3
    """Minimum number of columns required for substantial matches"""

    min_chunk_length: int = 2
    """Reject chunks shorter than this"""

    min_matched_rows: int = 10
    """Minimum matched rows for catalog match validation"""

    # Semantic anchor thresholds
    semantic_anchor_threshold: float = 2.0
    """Minimum semantic anchor score for substantial matches"""

    ambiguity_threshold: int = 5
    """Max number of tables with equal matches before rejecting as ambiguous"""

    # Confidence formula weights
    chunk_ratio_weight: float = 0.6
    """Weight for chunk ratio in base confidence calculation"""

    chunk_length_weight: float = 0.4
    """Weight for chunk length in base confidence calculation"""

    chunk_length_bonus_cap: float = 0.7
    """Maximum chunk length bonus"""

    chunk_length_bonus_base: float = 0.2
    """Base value for chunk length bonus calculation"""

    chunk_length_bonus_multiplier: float = 0.2
    """Multiplier for log2 in chunk length bonus"""

    anchor_boost_cap: float = 0.3
    """Maximum boost from semantic anchors (30%)"""

    anchor_boost_multiplier: float = 0.1
    """Multiplier for semantic anchor boost calculation"""

    # Overall confidence weights
    row_confidence_weight: float = 0.8
    """Weight for average row confidence in overall score"""

    match_rate_weight: float = 0.2
    """Weight for match rate in overall score"""


@dataclass
class SemanticAnchorWeights:
    """Weights for semantic anchor pattern matching.

    These weights are added to confidence scores when specific patterns
    are detected in the data (UUIDs, timestamps, URLs, etc.).
    """

    # Pattern detection weights
    uuid: float = 1.0
    """Weight for UUID pattern detection"""

    timestamp_text: float = 0.9
    """Weight for timestamp text pattern"""

    url: float = 0.8
    """Weight for URL pattern detection"""

    email: float = 0.7
    """Weight for email pattern detection"""

    domain: float = 0.6
    """Weight for domain pattern detection"""

    path: float = 0.5
    """Weight for file path pattern detection"""

    # Special match weights
    uuid_in_pk: float = 2.0
    """Weight for UUID found in primary keys"""

    most_common_high: float = 5.0
    """Weight for most common value with confidence >= 0.9"""

    most_common_high_threshold: float = 0.9
    """Confidence threshold for high most common value weight"""

    most_common_medium: float = 3.0
    """Weight for most common value with confidence >= 0.7"""

    most_common_medium_threshold: float = 0.7
    """Confidence threshold for medium most common value weight"""

    most_common_low: float = 1.5
    """Weight for most common value with confidence >= 0.5"""

    most_common_low_threshold: float = 0.5
    """Confidence threshold for low most common value weight"""

    example_value_text: float = 3.0
    """Weight for matching example text values"""

    example_value_numeric: float = 1.0
    """Weight for matching example numeric values"""

    enum_valid: float = 0.5
    """Weight for valid enum member detection"""

    # Foreign key weights
    fk_strong: float = 0.5
    """Weight for strong FK match (score >= 6, ratio >= 0.8)"""

    fk_strong_score_threshold: float = 6.0
    """Score threshold for strong FK weight"""

    fk_strong_ratio_threshold: float = 0.8
    """Ratio threshold for strong FK weight"""

    fk_medium: float = 0.3
    """Weight for medium FK match (score >= 5)"""

    fk_medium_score_threshold: float = 5.0
    """Score threshold for medium FK weight"""

    fk_weak: float = 0.2
    """Weight for weak FK match (score >= 4)"""

    fk_weak_score_threshold: float = 4.0
    """Score threshold for weak FK weight"""


@dataclass
class ScannerConfig:
    """Configuration for raw file scanning and recovery."""

    # Database limits
    max_db_size_mb: int = 1000
    """Maximum database size to process (in MB)"""

    connection_timeout: float = 5.0
    """SQLite connection timeout in seconds"""

    # Exemplar scanner settings
    max_workers: int = 1
    """Maximum number of parallel workers for exemplar scanning.

    IMPORTANT: Must be set to 1 for deterministic, reproducible results.
    Parallel processing (max_workers > 1) causes race conditions in file discovery,
    database processing order, and metadata writes, leading to inconsistent results
    across runs. Testing shows no performance benefit from parallelization, likely
    due to I/O bottlenecks and SQLite connection overhead.
    """

    min_non_null_threshold: int = 1
    """Minimum non-NULL rows required to consider database as having real data (vs NULL-only/empty)"""

    # Scanner behavior
    save_self_match_rubrics: bool = False
    """Save self-match rubrics for unidentified databases"""

    # File filtering
    ignore_files: set[str] = field(
        default_factory=lambda: {
            ".DS_Store",
            ".localized",
            ".Spotlight-V100",
            ".Trashes",
        }
    )
    """macOS metadata and system files to ignore during scanning"""

    ignore_prefixes: set[str] = field(default_factory=lambda: {"._"})
    """File name prefixes to ignore (e.g., AppleDouble resource forks)"""

    ignore_extensions: set[str] = field(
        default_factory=lambda: {
            # Images
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".tif",
            ".ico",
            ".webp",
            ".svg",
            ".heic",
            ".heif",
            ".raw",
            ".cr2",
            ".nef",
            ".dng",
            # Audio
            ".mp3",
            ".wav",
            ".flac",
            ".aac",
            ".m4a",
            ".ogg",
            ".wma",
            ".aiff",
            ".ape",
            ".opus",
            # Video
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
            ".mpg",
            ".mpeg",
            ".3gp",
            ".xar",
            # Documents (likely not databases)
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".odt",
            ".ods",
            ".odp",
            ".prproj",
            ".psd",
            # Archives (handled separately)
            ".gpg",
            ".dmg",
            ".pkg",
            ".iso",
            # Executables/Binaries
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".app",
            # Fonts
            ".ttf",
            ".otf",
            ".woff",
            ".woff2",
            # Other media
            ".swf",
            ".fla",
        }
    )
    """File extensions to skip during scanning (never databases/logs)"""


@dataclass
class OutputConfig:
    """Configuration for output structure and naming."""

    # Folder naming
    prefix: str = "MARS"
    """Prefix for output folder names"""

    timestamp_format: str = "%Y%m%d_%H%M%S"
    """Timestamp format for folder naming (strftime format)"""

    # Auto-naming behavior
    auto_timestamp: bool = True
    """Add timestamp to folder names automatically"""

    case_name_default: str = "Case"
    """Default case name when none is specified"""

    # Subdirectory names (customizable)
    reports_dir_name: str = "reports"
    """Name for reports directory"""

    databases_dir_name: str = "databases"
    """Name for databases directory"""

    schemas_dir_name: str = "schemas"
    """Name for schemas directory"""

    exports_dir_name: str = "exports"
    """Name for exports directory"""

    plots_dir_name: str = "plots"
    """Name for plots/visualizations directory"""

    def generate_folder_name(self, case_name: str | None = None) -> str:
        """Generate auto-named folder: MARS_Case_20240101_120000

        Args:
            case_name: Optional case name. Uses default if not provided.

        Returns:
            Formatted folder name string
        """
        case = case_name or self.case_name_default
        if self.auto_timestamp:
            timestamp = datetime.now(UTC).strftime(self.timestamp_format)
            return f"{self.prefix}_{case}_{timestamp}"
        return f"{self.prefix}_{case}"


@dataclass
class SuperRubricConfig:
    """Configuration for super-rubric generation and pooling.

    Super-rubrics are enriched rubrics generated from pooled data across
    multiple databases with matching schemas. They provide better matching
    quality for lost_and_found data by combining statistics from multiple
    data sources.
    """

    enabled: bool = True
    """Enable super-rubric framework (post-recovery generation for future runs)"""

    min_databases: int = 2
    """Minimum databases required to create super-rubric"""

    cache_dir: str = field(default_factory=lambda: str(Path(tempfile.gettempdir()) / "super_rubrics"))
    """Directory for caching super-rubrics and pooled databases"""


@dataclass
class SchemaComparisonConfig:
    """Configuration for database schema comparison and hashing.

    These settings control which tables are ignored when comparing database
    schemas or generating schema hashes for matching. This ensures consistent
    behavior between hash generation and runtime matching.
    """

    # Tables to ignore during schema comparison and hash generation
    ignorable_tables: set[str] = field(default_factory=lambda: GLOBAL_IGNORABLE_TABLES.copy())
    """Table names to ignore in schema comparison (metadata, stats, etc.)"""

    # Table name prefixes to ignore
    ignorable_prefixes: set[str] = field(default_factory=lambda: {"sqlite_", "sqlean_"})
    """Table name prefixes to ignore (SQLite internals, extensions)"""

    # Table name suffixes to ignore (FTS tables)
    ignorable_suffixes: set[str] = field(
        default_factory=lambda: {
            "_content",
            "_segments",
            "_segdir",
            "_docsize",
            "_stat",
        }
    )
    """Table name suffixes to ignore (Full-Text Search auxiliary tables)"""

    # Salvage/recovery table names to recognize
    salvage_tables: set[str] = field(
        default_factory=lambda: {
            "lost_and_found",
            "lostandfound",
            "_lost_and_found",
            "_lostandfound",
            "carved",
            "carved_rows",
            "recovered_rows",
        }
    )
    """Table names used for salvaged/recovered data (not part of original schema)"""


@dataclass
class VariantSelectorConfig:
    """Configuration for database variant selection.

    Controls how the variant selector chooses between Original/Clone/Recover/Dissect
    variants of SQLite databases during candidate processing.
    """

    dissect_all: bool = field(
        default=False,
        metadata={
            "user_configurable": True,
            "category": "advanced",
            "label": "Dissect All Variants",
            "description": "Run sqlite_dissect on all variants even without exemplar match",
        },
    )
    """Attempt sqlite_dissect on recovered variants even without exemplar match"""


@dataclass
class UIConfig:
    """Configuration for CLI/GUI behavior."""

    debug: bool = field(
        default=False,
        metadata={
            "user_configurable": True,
            "category": "basic",
            "label": "Debug Mode",
            "description": "Enable detailed diagnostic output",
        },
    )
    """Enable debug output (detailed diagnostic information)"""

    debug_log_to_file: bool = field(
        default=False,
        metadata={
            "user_configurable": True,
            "category": "basic",
            "label": "Save Debug to File",
            "description": "Save debug output to mars_debug.log in project folder",
        },
    )
    """Save debug output to log file (requires debug mode to be enabled)"""

    use_rich_output: bool = field(default=True, metadata={"user_configurable": False})
    """Use Rich library for formatted terminal output (internal)"""

    show_progress_bars: bool = field(
        default=True,
        metadata={
            "user_configurable": False,  # Controlled automatically by debug mode
            "category": "basic",
            "label": "Progress Bars",
            "description": "Show progress during processing (disabled when debug mode is on)",
        },
    )
    """Show progress bars during processing (automatically disabled when debug=True)"""


@dataclass
class CarverConfig:
    """Configuration for SQLite carving operations.

    Controls how SQLite databases are carved and processed, including
    timestamp filtering, protobuf decoding, and output formats.
    """

    ts_start: str = field(
        default="2015-01-01",
        metadata={
            "user_configurable": True,
            "category": "advanced",
            "label": "Timestamp Start",
            "description": "Tag timestamps only after this date (YYYY-MM-DD)",
            "validation": r"^\d{4}-\d{2}-\d{2}$",
        },
    )
    """Start date for timestamp filtering (YYYY-MM-DD format)"""

    ts_end: str = field(
        default="2030-01-01",
        metadata={
            "user_configurable": True,
            "category": "advanced",
            "label": "Timestamp End",
            "description": "Tag timestamps only before this date (YYYY-MM-DD)",
            "validation": r"^\d{4}-\d{2}-\d{2}$",
        },
    )
    """End date for timestamp filtering (YYYY-MM-DD format)"""

    filter_mode: str = field(
        default="permissive",
        metadata={
            "user_configurable": True,
            "category": "advanced",
            "label": "Timestamp Filter",
            "description": "Filtering mode: 'strict' (only confirmed), 'balanced' (+likely), 'permissive' (+ambiguous but no IDs), 'all'",
            "choices": ["permissive", "strict", "balanced", "all"],
        },
    )
    """Timestamp filtering mode: 'permissive', 'balanced', 'all', or 'strict'"""

    csv_export: bool = field(
        default=False,
        metadata={
            "user_configurable": True,
            "category": "advanced",
            "label": "Export CSV",
            "description": "Generate CSV output in addition to JSONL",
        },
    )
    """Enable CSV export alongside JSONL output"""

    decode_protobuf: bool = field(
        default=True,
        metadata={
            "user_configurable": True,
            "category": "advanced",
            "label": "Decode Protobuf",
            "description": "Attempt to decode protobuf data in BLOBs",
        },
    )
    """Attempt to decode protobuf data found in BLOB fields"""

    pretty_json: bool = field(
        default=True,
        metadata={
            "user_configurable": True,
            "category": "advanced",
            "label": "Pretty JSON",
            "description": "Format protobuf JSON output for readability",
        },
    )
    """Pretty-print JSON output for protobuf data"""

    enable_clustering: bool = field(default=True, metadata={"user_configurable": False})
    """Enable page clustering optimization (internal)"""

    # Expert-level options (not user-configurable via UI)
    parallel_processing: bool = field(default=True, metadata={"user_configurable": False})
    """Enable parallel page processing (expert option)"""

    parallel_threshold: int = field(default=10, metadata={"user_configurable": False})
    """Minimum pages required for parallel processing (expert option)"""

    max_workers: int | None = field(default=None, metadata={"user_configurable": False})
    """Number of worker threads for parallel processing (expert option, None = auto)"""


@dataclass
class ExemplarScanConfig:
    """Configuration for exemplar scanning operations.

    Controls timestamp validation ranges, rubric generation parameters,
    and which catalog groups are included in scans.
    """

    epoch_min: str = field(
        default="2000-01-01",
        metadata={
            "user_configurable": True,
            "category": "exemplar_timestamp",
            "special_handler": "_edit_exemplar_epoch_min",
            "label": "Epoch Minimum",
            "description": "Minimum date for valid timestamps",
            "validation": r"^\d{4}-\d{2}-\d{2}$",
        },
    )
    """Minimum date for timestamp validation (YYYY-MM-DD format)"""

    epoch_max: str = field(
        default="2038-01-19",
        metadata={
            "user_configurable": True,
            "category": "exemplar_timestamp",
            "special_handler": "_edit_exemplar_epoch_max",
            "label": "Epoch Maximum",
            "description": "Max date for valid timestamps",
            "validation": r"^\d{4}-\d{2}-\d{2}$",
        },
    )
    """Maximum date for timestamp validation (YYYY-MM-DD format)"""

    min_role_sample_size: int = field(
        default=5,
        metadata={
            "user_configurable": True,
            "category": "exemplar_rubric",
            "label": "Min Role Sample Size",
            "description": "Min rows needed for semantic role assignment",
            "validation": r"^\d+$",
            "min": 1,
            "max": 100,
        },
    )
    """Minimum non-null sample size for semantic role detection"""

    min_timestamp_rows: int = field(
        default=1,
        metadata={
            "user_configurable": True,
            "category": "exemplar_rubric",
            "label": "Min Timestamp Rows",
            "description": "Min timestamp values to assign timestamp role",
            "validation": r"^\d+$",
            "min": 1,
            "max": 100,
        },
    )
    """Minimum non-zero timestamp values required to assign timestamp role"""

    enabled_catalog_groups: list[str] = field(
        default_factory=list,
        metadata={
            "user_configurable": True,
            "category": "exemplar_catalog",
            "special_handler": "_edit_catalog_groups",
            "label": "Configure Groups",
            "description": "Catalog groups to include in scan",
        },
    )
    """Catalog groups to include in scan (empty list = all groups)"""

    excluded_file_types: list[str] = field(
        default_factory=list,
        metadata={
            "user_configurable": True,
            "category": "exemplar_filetype",
            "special_handler": "_edit_excluded_file_types",
            "label": "Excluded File Types",
            "description": "General file types to skip (e.g., 'cache', 'log')",
        },
    )
    """File types from catalog to exclude (e.g., 'cache' skips Firefox cache)"""


@dataclass
class MARSConfig:
    """Root configuration object containing all settings.

    This is the main configuration class that contains all subsections.
    It can be serialized to/from JSON for persistence.
    """

    matching: MatchingConfig = field(default_factory=MatchingConfig)
    """Matching and classification configuration"""

    semantic_anchors: SemanticAnchorWeights = field(default_factory=SemanticAnchorWeights)
    """Semantic anchor weights for pattern matching"""

    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    """Scanner and recovery configuration"""

    output: OutputConfig = field(default_factory=OutputConfig)
    """Output structure and naming configuration"""

    super_rubric: SuperRubricConfig = field(default_factory=SuperRubricConfig)
    """Super-rubric generation and pooling configuration"""

    schema_comparison: SchemaComparisonConfig = field(default_factory=SchemaComparisonConfig)
    """Schema comparison and hashing configuration"""

    variant_selector: VariantSelectorConfig = field(default_factory=VariantSelectorConfig)
    """Database variant selection configuration"""

    ui: UIConfig = field(default_factory=UIConfig)
    """UI and CLI behavior configuration"""

    carver: CarverConfig = field(default_factory=CarverConfig)
    """SQLite carving configuration"""

    exemplar: ExemplarScanConfig = field(default_factory=ExemplarScanConfig)
    """Exemplar scan configuration"""

    # Project metadata (populated from .marsproj file)
    project_name: str | None = None
    """Project name from .marsproj file"""

    examiner_name: str | None = None
    """Examiner name from .marsproj file"""

    case_number: str | None = None
    """Case number from .marsproj file"""

    description: str | None = None
    """Project description from .marsproj file"""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of the configuration
        """

        def convert_sets_to_lists(obj: Any) -> Any:
            """Recursively convert sets to lists for JSON serialization."""
            if isinstance(obj, dict):
                return {key: convert_sets_to_lists(value) for key, value in obj.items()}
            if isinstance(obj, set):
                return sorted(obj)  # Convert sets to sorted lists
            if isinstance(obj, list):
                return [convert_sets_to_lists(item) for item in obj]
            return obj

        return convert_sets_to_lists(asdict(self))  # type: ignore[return-value]

    def to_user_configurable_dict(self) -> dict[str, Any]:
        """Export only user-configurable settings (safe for user editing).

        This filters out all internal/expert settings that users shouldn't modify,
        keeping only fields marked with user_configurable=True.

        Returns:
            Dictionary with only user-configurable fields per section
        """
        result = {}

        # Process each config section
        for section_name in ["ui", "carver", "variant_selector", "exemplar"]:
            section_obj = getattr(self, section_name)
            section_class = type(section_obj)
            section_dict = {}

            # Extract only user_configurable fields
            for field_info in fields(section_class):
                if field_info.metadata.get("user_configurable", False):
                    field_value = getattr(section_obj, field_info.name)
                    section_dict[field_info.name] = field_value

            if section_dict:  # Only include section if it has user-configurable fields
                result[section_name] = section_dict

        return result

    def apply_user_configurable_dict(self, user_data: dict[str, Any]) -> None:
        """Apply user-configurable settings on top of current config.

        This safely applies only user-configurable fields, leaving all internal
        settings untouched.

        Args:
            user_data: Dictionary with user-configurable settings (from .marsproj)
        """
        # Apply settings to each section
        for section_name in ["ui", "carver", "variant_selector", "exemplar"]:
            if section_name not in user_data:
                continue

            section_obj = getattr(self, section_name)
            section_class = type(section_obj)
            section_data = user_data[section_name]

            # Only apply fields that are user_configurable
            for field_info in fields(section_class):
                if field_info.metadata.get("user_configurable", False) and field_info.name in section_data:
                    setattr(section_obj, field_info.name, section_data[field_info.name])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MARSConfig:
        """Create configuration from dictionary.

        Args:
            data: Dictionary containing configuration data

        Returns:
            MARSConfig instance
        """

        # Helper to convert lists back to sets for SchemaComparisonConfig
        def convert_lists_to_sets(config_dict: dict, set_fields: set[str]) -> dict:
            """Convert specified list fields back to sets."""
            result = config_dict.copy()
            for field_name in set_fields:
                if field_name in result and isinstance(result[field_name], list):
                    result[field_name] = set(result[field_name])
            return result

        # Nested dataclasses need special handling
        matching = MatchingConfig(**data.get("matching", {}))
        semantic_anchors = SemanticAnchorWeights(**data.get("semantic_anchors", {}))
        scanner = ScannerConfig(**data.get("scanner", {}))
        output = OutputConfig(**data.get("output", {}))
        super_rubric = SuperRubricConfig(**data.get("super_rubric", {}))

        # SchemaComparisonConfig has set fields that need conversion from lists
        schema_comparison_data = convert_lists_to_sets(
            data.get("schema_comparison", {}),
            {
                "ignorable_tables",
                "ignorable_prefixes",
                "ignorable_suffixes",
                "salvage_tables",
            },
        )
        schema_comparison = SchemaComparisonConfig(**schema_comparison_data)

        variant_selector = VariantSelectorConfig(**data.get("variant_selector", {}))
        ui = UIConfig(**data.get("ui", {}))
        carver = CarverConfig(**data.get("carver", {}))
        exemplar = ExemplarScanConfig(**data.get("exemplar", {}))

        return cls(
            matching=matching,
            semantic_anchors=semantic_anchors,
            scanner=scanner,
            output=output,
            super_rubric=super_rubric,
            schema_comparison=schema_comparison,
            variant_selector=variant_selector,
            ui=ui,
            carver=carver,
            exemplar=exemplar,
            project_name=data.get("project_name"),
            examiner_name=data.get("examiner_name"),
            case_number=data.get("case_number"),
            description=data.get("description"),
        )

    def __str__(self) -> str:
        """Return human-readable configuration summary."""
        lines = [
            "MARS Configuration",
            "=" * 50,
            f"Matching min_confidence: {self.matching.min_confidence}",
            f"Matching min_rows: {self.matching.min_rows}",
            f"Matching min_columns: {self.matching.min_columns}",
            f"Scanner max_db_size_mb: {self.scanner.max_db_size_mb}",
            f"Output prefix: {self.output.prefix}",
            f"Output auto_timestamp: {self.output.auto_timestamp}",
            f"UI debug: {self.ui.debug}",
        ]
        if self.project_name:
            lines.append(f"Project: {self.project_name}")
        if self.case_number:
            lines.append(f"Case: {self.case_number}")
        return "\n".join(lines)

    def should_ignore_file(self, file_path: Path) -> bool:
        """
        Check if a file should be ignored based on scanner configuration.

        Args:
            file_path: Path object to check

        Returns:
            True if file should be ignored, False otherwise

        Examples:
            >>> config = MARSConfig()
            >>> config.should_ignore_file(Path(".DS_Store"))
            True
            >>> config.should_ignore_file(Path("._resource_fork"))
            True
            >>> config.should_ignore_file(Path("photo.jpg"))
            True
            >>> config.should_ignore_file(Path("database.sqlite"))
            False
        """
        from pathlib import Path as PathClass

        if not isinstance(file_path, PathClass):
            file_path = PathClass(file_path)

        # Check exact filename matches
        if file_path.name in self.scanner.ignore_files:
            return True

        # Check prefixes
        if any(file_path.name.startswith(prefix) for prefix in self.scanner.ignore_prefixes):
            return True

        # Check extensions (case-insensitive)
        file_suffix = file_path.suffix.lower()
        return file_suffix in self.scanner.ignore_extensions
