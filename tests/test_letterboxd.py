import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lbxd.letterboxd import Film, load_export  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "export"


def test_reads_every_section():
    profile = load_export(FIXTURE)
    assert len(profile.watched) == 12
    assert len(profile.watchlist) == 1
    assert profile.watchlist[0].name == "The Godfather"


def test_diary_ratings_fill_gaps_without_clobbering():
    profile = load_export(FIXTURE)
    by_name = {f.name: f for f in profile.ratings}
    # ratings.csv has 3 entries; diary adds WALL-E and repeats The Matrix.
    assert len(profile.ratings) == 4
    assert by_name["WALL-E"].rating == 4.5
    assert by_name["The Matrix"].rating == 5.0


def test_seen_keys_are_case_insensitive_and_union_both_files():
    profile = load_export(FIXTURE)
    assert ("the matrix", 1999) in profile.seen_keys
    assert ("the godfather", 1972) not in profile.seen_keys  # watchlist only
    assert len(profile.seen_keys) == 12


def test_missing_and_malformed_fields_are_tolerated():
    assert Film("X", None).year is None
    assert Film("X", None).rating is None


def test_reads_a_zip_including_nested_paths(tmp_path):
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for csv_file in FIXTURE.glob("*.csv"):
            z.write(csv_file, f"letterboxd-export/{csv_file.name}")
    assert len(load_export(zip_path).watched) == 12


def test_rejects_unrelated_input(tmp_path):
    (tmp_path / "notes.csv").write_text("hello")
    with pytest.raises(ValueError, match="Letterboxd export"):
        load_export(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_export(tmp_path / "nope.zip")


def test_bom_prefixed_export_is_readable(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "e")
    watched = tmp_path / "e" / "watched.csv"
    watched.write_bytes(b"\xef\xbb\xbf" + watched.read_bytes())
    assert len(load_export(tmp_path / "e").watched) == 12
