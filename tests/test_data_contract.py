import unittest

from data_contract import REQUIRED_ARCHIVE_COLUMNS, validate_archive_columns


class ArchiveContractTests(unittest.TestCase):
    def test_accepts_the_current_archive_schema(self):
        validate_archive_columns(REQUIRED_ARCHIVE_COLUMNS)

    def test_reports_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "pl_eqt"):
            validate_archive_columns(REQUIRED_ARCHIVE_COLUMNS - {"pl_eqt"})


if __name__ == "__main__":
    unittest.main()
