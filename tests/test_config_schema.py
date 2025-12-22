"""Tests for configuration schema classes."""

from pathlib import Path

import pytest

from mars.config.schema import (
    GLOBAL_IGNORABLE_TABLES,
    MARSConfig,
    MatchingConfig,
    OutputConfig,
    ScannerConfig,
)


class TestGlobalIgnorableTables:
    """Test GLOBAL_IGNORABLE_TABLES constant."""

    def test_is_set(self):
        """Test that GLOBAL_IGNORABLE_TABLES is a set."""
        assert isinstance(GLOBAL_IGNORABLE_TABLES, set)

    def test_contains_sqlite_sequence(self):
        """Test that sqlite_sequence is in ignore list."""
        assert "sqlite_sequence" in GLOBAL_IGNORABLE_TABLES

    def test_contains_coredata_tables(self):
        """Test that CoreData tables are in ignore list."""
        assert "z_primarykey" in GLOBAL_IGNORABLE_TABLES
        assert "z_metadata" in GLOBAL_IGNORABLE_TABLES

    def test_contains_sqlean_define(self):
        """Test that sqlean_define is in ignore list."""
        assert "sqlean_define" in GLOBAL_IGNORABLE_TABLES


class TestMatchingConfig:
    """Test MatchingConfig defaults."""

    def test_default_values(self):
        """Test that MatchingConfig has expected defaults."""
        config = MatchingConfig()
        assert config.min_confidence == 0.7
        assert config.min_rows == 10
        assert config.min_columns == 3

    def test_custom_values(self):
        """Test MatchingConfig with custom values."""
        config = MatchingConfig(min_confidence=0.8, min_rows=5)
        assert config.min_confidence == 0.8
        assert config.min_rows == 5


class TestScannerConfig:
    """Test ScannerConfig defaults."""

    def test_default_ignore_files(self):
        """Test default ignore files list."""
        config = ScannerConfig()
        assert ".DS_Store" in config.ignore_files
        assert ".localized" in config.ignore_files

    def test_default_ignore_prefixes(self):
        """Test default ignore prefixes."""
        config = ScannerConfig()
        assert "._" in config.ignore_prefixes

    def test_default_ignore_extensions(self):
        """Test default ignore extensions."""
        config = ScannerConfig()
        assert ".jpg" in config.ignore_extensions
        assert ".mp4" in config.ignore_extensions
        assert ".pdf" in config.ignore_extensions
        assert ".png" in config.ignore_extensions
        assert ".gif" in config.ignore_extensions


class TestOutputConfig:
    """Test OutputConfig methods."""

    def test_default_prefix(self):
        """Test default output prefix."""
        config = OutputConfig()
        assert config.prefix == "MARS"

    def test_generate_folder_name_with_case(self):
        """Test folder name generation with case name."""
        config = OutputConfig()
        folder_name = config.generate_folder_name("TestCase")
        assert "MARS" in folder_name
        assert "TestCase" in folder_name

    def test_generate_folder_name_default(self):
        """Test folder name generation with default case name."""
        config = OutputConfig()
        folder_name = config.generate_folder_name()
        assert "MARS" in folder_name
        assert "Case" in folder_name  # Default case name

    def test_generate_folder_name_with_timestamp(self):
        """Test that generated folder names include timestamp when enabled."""
        config = OutputConfig(auto_timestamp=True)
        folder_name = config.generate_folder_name("Test")
        # Should have format MARS_Test_YYYYMMDD_HHMMSS
        assert len(folder_name) > len("MARS_Test")  # Has timestamp

    def test_generate_folder_name_without_timestamp(self):
        """Test folder name generation without timestamp."""
        config = OutputConfig(auto_timestamp=False)
        folder_name = config.generate_folder_name("Test")
        assert folder_name == "MARS_Test"


class TestMARSConfig:
    """Test MARSConfig class methods."""

    def test_default_creation(self):
        """Test creating MARSConfig with defaults."""
        config = MARSConfig()
        assert config.matching is not None
        assert config.scanner is not None
        assert config.output is not None

    def test_to_dict(self):
        """Test serialization to dict."""
        config = MARSConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "matching" in d
        assert "scanner" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        original = MARSConfig()
        original.matching.min_confidence = 0.9
        d = original.to_dict()

        restored = MARSConfig.from_dict(d)
        assert restored.matching.min_confidence == 0.9

    def test_round_trip(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = MARSConfig()
        original.matching.min_confidence = 0.85
        original.scanner.max_db_size_mb = 500

        d = original.to_dict()
        restored = MARSConfig.from_dict(d)

        assert restored.matching.min_confidence == original.matching.min_confidence
        assert restored.scanner.max_db_size_mb == original.scanner.max_db_size_mb


class TestMARSConfigShouldIgnoreFile:
    """Test MARSConfig.should_ignore_file method."""

    def test_ignore_ds_store(self):
        """Test that .DS_Store is ignored."""
        config = MARSConfig()
        assert config.should_ignore_file(Path(".DS_Store")) is True

    def test_ignore_apple_double(self):
        """Test that AppleDouble files (._*) are ignored."""
        config = MARSConfig()
        assert config.should_ignore_file(Path("._resource")) is True
        assert config.should_ignore_file(Path("._hidden")) is True

    def test_ignore_image_files(self):
        """Test that image files are ignored."""
        config = MARSConfig()
        assert config.should_ignore_file(Path("photo.jpg")) is True
        assert config.should_ignore_file(Path("image.PNG")) is True  # Case insensitive
        assert config.should_ignore_file(Path("screenshot.gif")) is True

    def test_ignore_video_files(self):
        """Test that video files are ignored."""
        config = MARSConfig()
        assert config.should_ignore_file(Path("video.mp4")) is True
        assert config.should_ignore_file(Path("movie.avi")) is True

    def test_dont_ignore_database_files(self):
        """Test that database files are not ignored."""
        config = MARSConfig()
        assert config.should_ignore_file(Path("database.db")) is False
        assert config.should_ignore_file(Path("cache.sqlite")) is False
        assert config.should_ignore_file(Path("data.sqlite3")) is False

    def test_dont_ignore_plist_files(self):
        """Test that plist files are not ignored."""
        config = MARSConfig()
        assert config.should_ignore_file(Path("prefs.plist")) is False

    def test_dont_ignore_log_files(self):
        """Test that log files are not ignored."""
        config = MARSConfig()
        assert config.should_ignore_file(Path("system.log")) is False

    def test_string_path_input(self):
        """Test that string paths are handled correctly."""
        config = MARSConfig()
        # Should accept string and convert to Path internally
        assert config.should_ignore_file(".DS_Store") is True  # type: ignore
        assert config.should_ignore_file("database.db") is False  # type: ignore


class TestMARSConfigStr:
    """Test MARSConfig string representation."""

    def test_str_representation(self):
        """Test that __str__ produces readable output."""
        config = MARSConfig()
        config.project_name = "Test Project"
        config.case_number = "12345"

        output = str(config)
        assert "MARS Configuration" in output
        assert "Test Project" in output
        assert "12345" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
