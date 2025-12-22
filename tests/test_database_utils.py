"""Tests for database utility functions."""

import sqlite3

import pytest

from mars.utils.database_utils import (
    get_chosen_variant_path,
    is_database_empty_or_null,
    is_encrypted_database,
    is_sqlite_database,
    quote_ident,
    quote_identifier,
)


class TestQuoteIdentifier:
    """Test SQL identifier quoting."""

    def test_simple_identifier(self):
        """Test quoting a simple identifier."""
        assert quote_identifier("users") == '"users"'
        assert quote_identifier("table_name") == '"table_name"'

    def test_identifier_with_embedded_quote(self):
        """Test escaping embedded double quotes."""
        assert quote_identifier('col"umn') == '"col""umn"'
        assert quote_identifier('table"with"quotes') == '"table""with""quotes"'

    def test_identifier_with_spaces(self):
        """Test quoting identifiers with spaces."""
        assert quote_identifier("table with spaces") == '"table with spaces"'
        assert quote_identifier("column name") == '"column name"'

    def test_identifier_with_special_chars(self):
        """Test quoting identifiers with special characters."""
        assert quote_identifier("table-name") == '"table-name"'
        assert quote_identifier("column.name") == '"column.name"'
        assert quote_identifier("table@name") == '"table@name"'

    def test_empty_identifier(self):
        """Test quoting empty identifier."""
        assert quote_identifier("") == '""'

    def test_quote_ident_alias(self):
        """Test that quote_ident is an alias for quote_identifier."""
        assert quote_ident("users") == quote_identifier("users")
        assert quote_ident('col"umn') == quote_identifier('col"umn')


class TestIsSqliteDatabase:
    """Test SQLite database validation."""

    def test_valid_sqlite_database(self, tmp_path):
        """Test detection of valid SQLite database."""
        db_path = tmp_path / "valid.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'Alice')")
        conn.commit()
        conn.close()

        assert is_sqlite_database(db_path) is True

    def test_empty_sqlite_database(self, tmp_path):
        """Test detection of empty but valid SQLite database."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        # Create a table so the database has proper structure
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        assert is_sqlite_database(db_path) is True

    def test_non_sqlite_file(self, tmp_path):
        """Test rejection of non-SQLite file."""
        bad_file = tmp_path / "bad.db"
        bad_file.write_bytes(b"not a sqlite database")

        assert is_sqlite_database(bad_file) is False

    def test_random_binary_file(self, tmp_path):
        """Test rejection of random binary data."""
        random_file = tmp_path / "random.bin"
        random_file.write_bytes(bytes(range(256)))

        assert is_sqlite_database(random_file) is False

    def test_nonexistent_file(self, tmp_path):
        """Test handling of nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.db"

        assert is_sqlite_database(nonexistent) is False

    def test_text_file(self, tmp_path):
        """Test rejection of text file."""
        text_file = tmp_path / "text.txt"
        text_file.write_text("Hello World")

        assert is_sqlite_database(text_file) is False

    def test_sqlite_magic_bytes(self, tmp_path):
        """Test detection of SQLite magic bytes."""
        # Create file with correct header but invalid content
        fake_sqlite = tmp_path / "fake.db"
        fake_sqlite.write_bytes(b"SQLite format 3\x00" + b"\x00" * 100)

        # Should fail because it can't be queried
        assert is_sqlite_database(fake_sqlite) is False


class TestIsEncryptedDatabase:
    """Test encrypted database detection."""

    def test_valid_sqlite_not_encrypted(self, tmp_path):
        """Test that valid SQLite is not flagged as encrypted."""
        db_path = tmp_path / "valid.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        assert is_encrypted_database(db_path) is False

    def test_non_database_extension_not_flagged(self, tmp_path):
        """Test that non-database files are not flagged as encrypted."""
        # Random binary file without database extension
        random_file = tmp_path / "random.bin"
        random_file.write_bytes(bytes(range(256)))

        assert is_encrypted_database(random_file) is False

    def test_log_file_not_flagged(self, tmp_path):
        """Test that log files are not flagged as encrypted."""
        log_file = tmp_path / "application.log.db"
        log_file.write_bytes(bytes(range(256)))

        assert is_encrypted_database(log_file) is False

    def test_plist_file_not_flagged(self, tmp_path):
        """Test that plist files are not flagged as encrypted."""
        plist_file = tmp_path / "preferences.plist"
        plist_file.write_bytes(bytes(range(256)))

        assert is_encrypted_database(plist_file) is False

    def test_gzip_file_not_flagged(self, tmp_path):
        """Test that gzip files are not flagged as encrypted."""
        gzip_file = tmp_path / "archive.sqlite"
        gzip_file.write_bytes(b"\x1f\x8b" + bytes(range(254)))

        assert is_encrypted_database(gzip_file) is False

    def test_zip_file_not_flagged(self, tmp_path):
        """Test that ZIP files are not flagged as encrypted."""
        zip_file = tmp_path / "archive.sqlite"
        zip_file.write_bytes(b"\x50\x4b\x03\x04" + bytes(range(252)))

        assert is_encrypted_database(zip_file) is False


class TestIsDatabaseEmptyOrNull:
    """Test empty database detection."""

    def test_database_with_data(self, tmp_path):
        """Test detection of database with actual data."""
        db_path = tmp_path / "with_data.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.commit()
        conn.close()

        is_empty, reason = is_database_empty_or_null(db_path)
        assert is_empty is False
        assert reason == ""

    def test_database_no_tables(self, tmp_path):
        """Test detection of database with no tables."""
        db_path = tmp_path / "no_tables.db"
        conn = sqlite3.connect(db_path)
        conn.close()

        is_empty, reason = is_database_empty_or_null(db_path)
        assert is_empty is True
        assert reason == "no_tables"

    def test_database_empty_table(self, tmp_path):
        """Test detection of database with empty table."""
        db_path = tmp_path / "empty_table.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.commit()
        conn.close()

        is_empty, reason = is_database_empty_or_null(db_path)
        assert is_empty is True
        assert reason == "no_rows"

    def test_database_all_nulls(self, tmp_path):
        """Test detection of database with only NULL values."""
        db_path = tmp_path / "all_null.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (NULL, NULL)")
        conn.execute("INSERT INTO users VALUES (NULL, NULL)")
        conn.commit()
        conn.close()

        is_empty, reason = is_database_empty_or_null(db_path)
        assert is_empty is True
        assert reason == "all_null"


class TestGetChosenVariantPath:
    """Test variant path extraction from records."""

    def test_original_variant(self, tmp_path):
        """Test extraction of original variant path."""
        db_path = tmp_path / "original.sqlite"
        db_path.touch()

        record = {
            "variant_chosen": "O",
            "variant_outputs": {"original": str(db_path)},
        }
        result = get_chosen_variant_path(record)
        assert result == db_path

    def test_clone_variant(self, tmp_path):
        """Test extraction of clone variant path."""
        db_path = tmp_path / "clone.sqlite"
        db_path.touch()

        record = {
            "variant_chosen": "C",
            "variant_outputs": {"clone": str(db_path)},
        }
        result = get_chosen_variant_path(record)
        assert result == db_path

    def test_recover_variant(self, tmp_path):
        """Test extraction of recover variant path."""
        db_path = tmp_path / "recover.sqlite"
        db_path.touch()

        record = {
            "variant_chosen": "R",
            "variant_outputs": {"recover": str(db_path)},
        }
        result = get_chosen_variant_path(record)
        assert result == db_path

    def test_dissect_variant(self, tmp_path):
        """Test extraction of dissect variant path."""
        db_path = tmp_path / "dissect.sqlite"
        db_path.touch()

        record = {
            "variant_chosen": "D",
            "variant_outputs": {"dissect_rebuilt": str(db_path)},
        }
        result = get_chosen_variant_path(record)
        assert result == db_path

    def test_decompressed_variant_preferred(self, tmp_path):
        """Test that decompressed path is preferred over other variants."""
        decompressed_path = tmp_path / "decompressed.sqlite"
        original_path = tmp_path / "original.sqlite"
        decompressed_path.touch()
        original_path.touch()

        record = {
            "variant_chosen": "O",
            "variant_outputs": {
                "original": str(original_path),
                "decompressed": str(decompressed_path),
            },
        }
        result = get_chosen_variant_path(record)
        assert result == decompressed_path

    def test_missing_variant_outputs(self):
        """Test handling of missing variant_outputs."""
        record = {"variant_chosen": "O"}
        result = get_chosen_variant_path(record)
        assert result is None

    def test_nonexistent_variant_path(self, tmp_path):
        """Test handling of nonexistent variant path."""
        record = {
            "variant_chosen": "O",
            "variant_outputs": {"original": str(tmp_path / "nonexistent.sqlite")},
        }
        result = get_chosen_variant_path(record)
        assert result is None

    def test_unknown_variant_tag(self, tmp_path):
        """Test handling of unknown variant tag."""
        db_path = tmp_path / "unknown.sqlite"
        db_path.touch()

        record = {
            "variant_chosen": "Z",  # Unknown tag
            "variant_outputs": {"original": str(db_path)},
        }
        result = get_chosen_variant_path(record)
        assert result is None

    def test_empty_record(self):
        """Test handling of empty record."""
        result = get_chosen_variant_path({})
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
