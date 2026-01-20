"""Path management with auto-naming for MARS.

Provides unified path structure for all output directories with
automatic folder naming based on configuration.
"""

from dataclasses import dataclass
from pathlib import Path

from mars.config.schema import OutputConfig


@dataclass
class ProjectPaths:
    """Manage all project paths with auto-naming.

    This is the single source of truth for all output directories.
    Supports both exemplar scanning and candidate processing workflows.
    """

    # Root directory
    root: Path
    """Root project directory"""

    # Top-level directories
    reports: Path
    """Reports and analysis output"""

    plots: Path
    """Interactive plots and charts (plotly output)"""

    databases: Path
    """All database-related output"""

    exports: Path
    """Exported data (CSV, JSON, etc.)"""

    temp: Path
    """Temporary files during processing"""

    # Exemplar scan directories (for cataloging known databases)
    exemplar: Path
    """Exemplar scan root directory"""

    exemplar_databases: Path
    """Exemplar databases parent directory"""

    exemplar_catalog: Path
    """Cataloged databases from exemplar scan"""

    exemplar_originals: Path
    """Original exemplar database files"""

    exemplar_encrypted: Path
    """Encrypted database files from exemplar scan"""

    exemplar_schemas: Path
    """Schemas extracted from exemplar databases"""

    exemplar_logs: Path
    """Non-database logs from exemplar scan"""

    exemplar_caches: Path
    """Cache files from exemplar scan"""

    exemplar_keychains: Path
    """Keychain files from exemplar scan"""

    # Candidate scan directories
    candidates: Path
    """Root directory for candidate scan runs"""

    # Imported exemplar packages directory
    imports: Path
    """Directory for imported exemplar packages"""

    # Exported exemplar packages directory
    package_exports: Path
    """Directory for exported exemplar packages"""

    # Candidate database subdirectories (per-run paths)
    db_selected_variants: Path
    """Individual database variant folders (under databases/)"""

    db_catalog: Path
    """Exemplar-matched databases (identified/known)"""

    db_metamatches: Path
    """Metamatch-grouped databases (unknown but schema-similar)"""

    db_schemas: Path
    """Schema files for metamatch databases (exemplar schemas in exemplar/)"""

    db_carved: Path
    """Carved databases (forensic carving output)"""

    db_encrypted: Path
    """Encrypted databases that cannot be processed"""

    # Artifact directories
    logs: Path
    """Text log files (wifi.log, system.log, etc.)"""

    caches: Path
    """Cache files (Firefox cache2, Safari cache, etc.)"""

    keychains: Path
    """Keychain files"""

    @staticmethod
    def _build_candidate_db_paths(databases_dir: Path) -> dict[str, Path]:
        """Build candidate database subdirectory paths.

        Single source of truth for candidate database path structure.
        """
        return {
            "db_selected_variants": databases_dir / "selected_variants",
            "db_catalog": databases_dir / "catalog",
            "db_metamatches": databases_dir / "metamatches",
            "db_schemas": databases_dir / "schemas",
            "db_carved": databases_dir / "carved",
            "db_encrypted": databases_dir / "encrypted",
        }

    @staticmethod
    def _build_exemplar_paths(root: Path) -> dict[str, Path]:
        """Build exemplar scan directory paths.

        Single source of truth for exemplar path structure.
        """
        exemplar = root / "exemplar"
        exemplar_databases = exemplar / "databases"
        return {
            "exemplar": exemplar,
            "exemplar_databases": exemplar_databases,
            "exemplar_catalog": exemplar_databases / "catalog",
            "exemplar_originals": exemplar_databases / "originals",
            "exemplar_encrypted": exemplar_databases / "encrypted",
            "exemplar_schemas": exemplar_databases / "schemas",
            "exemplar_logs": exemplar / "logs",
            "exemplar_caches": exemplar / "caches",
            "exemplar_keychains": exemplar / "keychains",
        }

    @classmethod
    def create_candidate_run(
        cls,
        candidates_root: Path,
        run_name: str,
        output_config: OutputConfig | None = None,
    ) -> "ProjectPaths":
        """Create ProjectPaths for a candidate run under candidates directory.

        Args:
            candidates_root: Root candidates directory (e.g., /output/MARS_{project_name}_{timestamp}/candidates)
            run_name: Name for this candidate run (timestamp)
            output_config: Optional output configuration (uses defaults if None)

        Returns:
            ProjectPaths instance scoped to this candidate run
        """
        if output_config is None:
            output_config = OutputConfig()

        # Candidate run root
        run_root = candidates_root / run_name

        # Top-level directories
        databases = run_root / "databases"
        reports = run_root / "reports"
        plots = reports / "plots"
        exports = run_root / "exports"
        temp = run_root / "_temp"

        # Artifact type directories
        logs = run_root / "logs"
        caches = run_root / "caches"
        keychains = run_root / "keychains"

        # Build candidate database subdirectories using helper
        db_paths = cls._build_candidate_db_paths(databases)

        # Set unused paths to temp (they won't be created)
        unused = temp / "_unused"

        # Build exemplar paths (unused for candidate runs, but required by dataclass)
        exemplar_paths = cls._build_exemplar_paths(unused)

        return cls(
            root=run_root,
            reports=reports,
            plots=plots,
            databases=databases,
            exports=exports,
            temp=temp,
            logs=logs,
            caches=caches,
            keychains=keychains,
            # Exemplar paths (unused in candidate runs)
            exemplar=exemplar_paths["exemplar"],
            exemplar_databases=exemplar_paths["exemplar_databases"],
            exemplar_catalog=exemplar_paths["exemplar_catalog"],
            exemplar_originals=exemplar_paths["exemplar_originals"],
            exemplar_encrypted=exemplar_paths["exemplar_encrypted"],
            exemplar_schemas=exemplar_paths["exemplar_schemas"],
            exemplar_logs=exemplar_paths["exemplar_logs"],
            exemplar_caches=exemplar_paths["exemplar_caches"],
            exemplar_keychains=exemplar_paths["exemplar_keychains"],
            # Candidates root (parent directory for this run)
            candidates=candidates_root,
            # Imports directory (not used in candidate runs)
            imports=unused / "imports",
            # Package exports directory (not used in candidate runs)
            package_exports=unused / "exports",
            # Candidate database subdirectories
            db_selected_variants=db_paths["db_selected_variants"],
            db_catalog=db_paths["db_catalog"],
            db_metamatches=db_paths["db_metamatches"],
            db_schemas=db_paths["db_schemas"],
            db_carved=db_paths["db_carved"],
            db_encrypted=db_paths["db_encrypted"],
        )

    @classmethod
    def from_existing(
        cls,
        root: Path,
        output_config: OutputConfig | None = None,
    ) -> "ProjectPaths":
        """Reconstruct ProjectPaths from an existing root directory.

        Args:
            root: Existing root directory
            output_config: Optional output configuration (uses defaults if None)

        Returns:
            ProjectPaths instance with all paths configured
        """

        if output_config is None:
            output_config = OutputConfig()

        # Top-level directories
        databases = root / output_config.databases_dir_name
        reports = root / output_config.reports_dir_name
        plots = reports / "plots"
        exports = root / output_config.exports_dir_name
        temp = root / "_temp"

        # Artifact type directories
        logs = root / "logs"
        caches = root / "caches"
        keychains = root / "keychains"

        # Build exemplar paths using helper
        exemplar_paths = cls._build_exemplar_paths(root)

        # Candidate-specific paths not used in main project structure
        unused = temp / "_unused"
        db_paths = cls._build_candidate_db_paths(unused)

        return cls(
            root=root,
            reports=reports,
            plots=plots,
            databases=databases,
            exports=exports,
            temp=temp,
            logs=logs,
            caches=caches,
            keychains=keychains,
            # Exemplar paths
            exemplar=exemplar_paths["exemplar"],
            exemplar_databases=exemplar_paths["exemplar_databases"],
            exemplar_catalog=exemplar_paths["exemplar_catalog"],
            exemplar_originals=exemplar_paths["exemplar_originals"],
            exemplar_encrypted=exemplar_paths["exemplar_encrypted"],
            exemplar_schemas=exemplar_paths["exemplar_schemas"],
            exemplar_logs=exemplar_paths["exemplar_logs"],
            exemplar_caches=exemplar_paths["exemplar_caches"],
            exemplar_keychains=exemplar_paths["exemplar_keychains"],
            # Candidates directory
            candidates=root / "candidates",
            # Imports directory for external exemplar packages
            imports=root / "imports",
            # Package exports directory for exported exemplar packages
            package_exports=root / "exports",
            # Candidate-specific paths (unused in main project structure)
            db_selected_variants=db_paths["db_selected_variants"],
            db_catalog=db_paths["db_catalog"],
            db_metamatches=db_paths["db_metamatches"],
            db_schemas=db_paths["db_schemas"],
            db_carved=db_paths["db_carved"],
            db_encrypted=db_paths["db_encrypted"],
        )

    @classmethod
    def create(
        cls,
        base_dir: Path,
        output_config: OutputConfig,
        case_name: str | None = None,
    ) -> "ProjectPaths":
        """Create project paths with auto-generated folder name.

        Args:
            base_dir: Base directory for output
            output_config: Output configuration for naming
            case_name: Optional case name (uses default from config if None)

        Returns:
            ProjectPaths instance with all paths configured
        """
        # Generate auto-named folder: MARS_Case_20240101_120000
        folder_name = output_config.generate_folder_name(case_name)
        root = base_dir / folder_name

        # Top-level directories
        databases = root / output_config.databases_dir_name
        reports = root / output_config.reports_dir_name
        plots = reports / "plots"
        exports = root / output_config.exports_dir_name
        temp = root / "_temp"

        # Artifact type directories
        logs = root / "logs"
        caches = root / "caches"
        keychains = root / "keychains"

        # Build exemplar paths using helper
        exemplar_paths = cls._build_exemplar_paths(root)

        # Candidate-specific paths (not used in main project structure)
        unused = temp / "_unused"
        db_paths = cls._build_candidate_db_paths(unused)

        return cls(
            root=root,
            reports=reports,
            plots=plots,
            databases=databases,
            exports=exports,
            temp=temp,
            logs=logs,
            caches=caches,
            keychains=keychains,
            # Exemplar paths
            exemplar=exemplar_paths["exemplar"],
            exemplar_databases=exemplar_paths["exemplar_databases"],
            exemplar_catalog=exemplar_paths["exemplar_catalog"],
            exemplar_originals=exemplar_paths["exemplar_originals"],
            exemplar_encrypted=exemplar_paths["exemplar_encrypted"],
            exemplar_schemas=exemplar_paths["exemplar_schemas"],
            exemplar_logs=exemplar_paths["exemplar_logs"],
            exemplar_caches=exemplar_paths["exemplar_caches"],
            exemplar_keychains=exemplar_paths["exemplar_keychains"],
            # Candidates directory
            candidates=root / "candidates",
            # Imports directory for external exemplar packages
            imports=root / "imports",
            # Package exports directory for exported exemplar packages
            package_exports=root / "exports",
            # Candidate-specific paths (unused in main project structure)
            db_selected_variants=db_paths["db_selected_variants"],
            db_catalog=db_paths["db_catalog"],
            db_metamatches=db_paths["db_metamatches"],
            db_schemas=db_paths["db_schemas"],
            db_carved=db_paths["db_carved"],
            db_encrypted=db_paths["db_encrypted"],
        )

    def create_all(self) -> None:
        """Create all directories in the project structure.

        Creates all directories with parents, existing directories are not modified.

        Raises:
            OSError: If directories cannot be created
        """
        for path in self._all_paths():
            path.mkdir(parents=True, exist_ok=True)

    def create_exemplar_dirs(self) -> None:
        """Create only directories needed for exemplar scanning workflow.

        Exemplar scans catalog known databases and extract schemas.
        They need: databases, reports, logs, caches, keychains.

        Raises:
            OSError: If directories cannot be created
        """
        paths_to_create = [
            self.root,
            self.reports,
            self.exemplar,
            self.exemplar_databases,
            self.exemplar_catalog,
            self.exemplar_originals,
            self.exemplar_schemas,
            self.exemplar_logs,
            self.exemplar_caches,
            self.exemplar_keychains,
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def create_candidate_dirs(self) -> None:
        """Create only directories needed for candidate processing workflow.

        Raises:
            OSError: If directories cannot be created
        """
        paths_to_create = [
            self.root,
            self.databases,
            self.db_selected_variants,
            self.db_catalog,
            self.db_metamatches,
            self.db_schemas,
            self.logs,
            self.caches,
            self.reports,
            self.exports,
            self.temp,
        ]
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)

    def _all_paths(self) -> list[Path]:
        """Get list of all paths for iteration.

        Returns:
            List of all Path objects in this structure
        """
        return [
            self.root,
            self.reports,
            self.databases,
            self.exports,
            self.temp,
            self.logs,
            self.caches,
            self.keychains,
            self.exemplar,
            self.exemplar_databases,
            self.exemplar_catalog,
            self.exemplar_originals,
            self.exemplar_schemas,
            self.exemplar_logs,
            self.exemplar_caches,
            self.exemplar_keychains,
            self.candidates,
            self.imports,
            self.db_selected_variants,
            self.db_catalog,
            self.db_metamatches,
            self.db_schemas,
            self.db_encrypted,
        ]

    def clean_temp(self) -> None:
        """Remove temporary directory and all contents.

        Raises:
            OSError: If directory cannot be removed
        """
        from mars.utils.cleanup_utilities import cleanup_sqlite_directory

        if self.temp.exists():
            cleanup_sqlite_directory(self.temp)

    def exists(self) -> bool:
        """Check if root directory exists.

        Returns:
            True if root directory exists
        """
        return self.root.exists()
