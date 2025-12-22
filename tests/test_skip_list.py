"""Tests for database skip list functionality."""

import pytest

from mars.pipeline.common.catalog_manager import CatalogManager


@pytest.fixture
def catalog_manager():
    """Create a CatalogManager instance for testing."""
    return CatalogManager()


class TestCatalogLoading:
    """Test catalog loading functionality."""

    def test_catalog_loads_successfully(self, catalog_manager):
        """Test that catalog loads without errors."""
        catalog = catalog_manager.get_catalog()
        assert catalog is not None, "Catalog should load successfully"
        assert isinstance(catalog, dict), "Catalog should be a dictionary"

    def test_catalog_has_expected_keys(self, catalog_manager):
        """Test that catalog contains expected top-level keys."""
        catalog = catalog_manager.get_catalog()
        # Should have skip_databases and browser categories
        assert "skip_databases" in catalog
        assert "safari" in catalog or "chrome" in catalog


class TestSkipDatabasesLoading:
    """Test skip_databases configuration loading."""

    def test_skip_databases_loads(self, catalog_manager):
        """Test that skip_databases section loads from catalog."""
        skip_dbs = catalog_manager.get_skip_databases()
        assert skip_dbs is not None, "Skip databases should load"
        assert isinstance(skip_dbs, dict), "Skip databases should be a dictionary"

    def test_geoservices_in_skip_list(self, catalog_manager):
        """Test that geoservices is configured in skip list."""
        skip_dbs = catalog_manager.get_skip_databases()
        assert "geoservices" in skip_dbs, "GeoServices should be in skip list"

    def test_geoservices_has_required_fields(self, catalog_manager):
        """Test that geoservices skip entry has required fields."""
        skip_dbs = catalog_manager.get_skip_databases()
        geo = skip_dbs["geoservices"]

        assert "name" in geo
        assert "reason" in geo
        assert "table_patterns" in geo
        assert isinstance(geo["table_patterns"], list)
        assert len(geo["table_patterns"]) > 0


class TestSkipDetection:
    """Test skip detection logic."""

    def test_geoservices_detected_correctly(self, catalog_manager):
        """Test that GeoServices database is correctly identified."""
        # Typical GeoServices tables
        geoservices_tables = [
            "GeoPlaces",
            "GeoPlaces_RTree_node",
            "GeoPlaces_RTree_rowid",
            "GeoPlaceNames",
            "GeoLookup_da",
            "GeoLookup_en",
            "GeoLookup_es",
            "GeoLookup_fr",
            "GeoLookup_de",
            "GeoLookup_it",
            "RKTimeZone",
            "RKTimeZone_RTree_node",
        ]

        should_skip, category, reason = catalog_manager.should_skip_database(geoservices_tables)

        assert should_skip is True, "GeoServices should be skipped"
        assert category == "geoservices", f"Category should be 'geoservices', got '{category}'"
        assert reason != "", "Should have a reason for skipping"
        assert "cache" in reason.lower() or "system" in reason.lower()

    def test_normal_database_not_skipped(self, catalog_manager):
        """Test that normal databases are not incorrectly skipped."""
        # Typical browser history tables
        normal_tables = ["history_visits", "history_items", "urls", "visits"]

        should_skip, category, reason = catalog_manager.should_skip_database(normal_tables)

        assert should_skip is False, "Normal database should NOT be skipped"
        assert category == "", "Category should be empty for non-skipped databases"
        assert reason == "", "Reason should be empty for non-skipped databases"

    def test_partial_match_not_skipped(self, catalog_manager):
        """Test that databases with only 1-2 matching tables are not skipped."""
        # Only has 2 GeoServices tables - shouldn't skip (requires 3+)
        partial_tables = ["GeoPlaces", "GeoPlaceNames", "other_table", "another_table"]

        should_skip, _, _ = catalog_manager.should_skip_database(partial_tables)

        # Should NOT skip with only 2 matches (requires 3+ for safety)
        assert should_skip is False, "Partial matches should not trigger skip"

    def test_wildcard_pattern_matching(self, catalog_manager):
        """Test that wildcard patterns (e.g., GeoLookup_*) work correctly."""
        # Multiple GeoLookup_* tables should trigger skip
        wildcard_tables = [
            "GeoLookup_en",
            "GeoLookup_fr",
            "GeoLookup_de",
            "GeoLookup_es",
            "GeoLookup_it",
            "other_table",
        ]

        should_skip, category, _ = catalog_manager.should_skip_database(wildcard_tables)

        assert should_skip is True, "Wildcard pattern should match"
        assert category == "geoservices"

    def test_empty_table_list_not_skipped(self, catalog_manager):
        """Test that empty table list doesn't cause errors."""
        should_skip, category, reason = catalog_manager.should_skip_database([])

        assert should_skip is False
        assert category == ""
        assert reason == ""

    def test_no_catalog_graceful_failure(self, catalog_manager):
        """Test that missing catalog doesn't crash skip detection."""
        # Clear the catalog cache to simulate missing catalog
        catalog_manager._catalog_cache = None

        # Mock the catalog path to not exist
        original_path = catalog_manager.catalog_path
        from pathlib import Path

        catalog_manager.catalog_path = Path("/nonexistent/catalog.yaml")

        should_skip, _, _ = catalog_manager.should_skip_database(["any_table"])

        # Should not crash, just return False
        assert should_skip is False

        # Restore
        catalog_manager.catalog_path = original_path


class TestSkipListExtensibility:
    """Test that skip list can be extended with more databases."""

    def test_multiple_skip_categories_supported(self, catalog_manager):
        """Test that multiple skip categories can coexist."""
        skip_dbs = catalog_manager.get_skip_databases()

        # Currently we have geoservices
        assert len(skip_dbs) >= 1

        # Should be able to have multiple categories
        # (This test will pass even if only geoservices exists,
        # but documents expected behavior for future additions)
        for category, info in skip_dbs.items():
            assert "name" in info, f"Category {category} should have name"
            assert "table_patterns" in info, f"Category {category} should have patterns"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
