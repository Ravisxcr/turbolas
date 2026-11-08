"""Unit tests for in-memory columnar ASCII numeric parser."""

from __future__ import annotations

import math
import struct

import pytest
from turbolas import (
    ColumnarBuffer,
    ColumnarParser,
    LASDataError,
    NullSentinel,
    decode_buffer,
    estimate_row_capacity,
    parse_columnar,
)


def test_estimate_row_capacity() -> None:
    assert estimate_row_capacity(0, 4) == 0
    assert estimate_row_capacity(1000, 0) == 0
    cap = estimate_row_capacity(100_000, 4)
    assert cap >= 2500


def test_parse_basic_unwrapped_bytes_and_str() -> None:
    raw = b" 1000.0   25.5   120.3\n 1000.5   26.0   121.0\n 1001.0   27.2   122.5\n"
    names = ["DEPT", "GR", "RHOB"]

    # From bytes
    buf = parse_columnar(raw, num_columns=3, column_names=names)
    assert isinstance(buf, ColumnarBuffer)
    assert buf.num_rows == 3
    assert buf.num_columns == 3
    assert buf.shape == (3, 3)
    assert buf.column_names == names
    assert not buf.is_empty
    assert len(buf) == 3

    assert buf.column(0) == [1000.0, 1000.5, 1001.0]
    assert buf.column("DEPT") == [1000.0, 1000.5, 1001.0]
    assert buf["GR"] == [25.5, 26.0, 27.2]
    assert buf[-1] == [120.3, 121.0, 122.5]
    assert buf.row(0) == [1000.0, 25.5, 120.3]
    assert buf.cell(1, "GR") == 26.0
    assert "GR" in buf
    assert "UNKNOWN" not in buf

    # From str
    text = raw.decode("ascii")
    buf_str = parse_columnar(text, num_columns=3, column_names=names)
    assert buf_str.column(1) == [25.5, 26.0, 27.2]


def test_parse_from_decoded_buffer() -> None:
    raw = b" 100.0  200.0\n 101.0  201.0\n"
    decoded = decode_buffer(raw)
    parser = ColumnarParser(num_columns=2, column_names=["A", "B"])
    buf = parser.parse(decoded)
    assert buf.num_rows == 2
    assert buf["A"] == [100.0, 101.0]
    assert buf["B"] == [200.0, 201.0]


def test_scientific_notation_and_signs() -> None:
    data = b" 1.234e-03  -4.56E+02  +0.0\n -1.5e2  .75  -.25\n"
    buf = parse_columnar(data, num_columns=3)
    assert buf.num_rows == 2
    assert pytest.approx(buf.column(0)[0]) == 1.234e-3
    assert buf.column(1)[0] == -456.0
    assert buf.column(2)[0] == 0.0
    assert buf.column(0)[1] == -150.0
    assert buf.column(1)[1] == 0.75
    assert buf.column(2)[1] == -0.25


def test_null_sentinel_mapping() -> None:
    data = b" 1000.0  -999.25  50.0\n 1000.5  NaN  null\n 1001.0  None  -nan\n"
    sentinel = NullSentinel.default_cwls()
    buf = parse_columnar(data, num_columns=3, null_sentinel=sentinel)

    assert buf.num_rows == 3
    assert buf.column(0) == [1000.0, 1000.5, 1001.0]

    col1 = buf.column(1)
    assert math.isnan(col1[0])
    assert math.isnan(col1[1])
    assert math.isnan(col1[2])
    assert buf.null_count(1) == 3

    col2 = buf.column(2)
    assert col2[0] == 50.0
    assert math.isnan(col2[1])
    assert math.isnan(col2[2])
    assert buf.null_count(2) == 2


def test_custom_null_sentinel() -> None:
    data = b" 100.0 -9999.0\n 101.0 45.0\n"
    sentinel = NullSentinel(-9999.0)
    buf = parse_columnar(data, num_columns=2, null_sentinel=sentinel)
    assert buf.num_rows == 2
    assert math.isnan(buf.column(1)[0])
    assert buf.column(1)[1] == 45.0


def test_wrapped_mode() -> None:
    # 5 curves wrapped across 2 lines per depth step
    data = b" 1000.0 25.5 120.3\n 0.45 2.65\n 1000.5 26.0 121.0\n 0.46 2.66\n"
    names = ["DEPT", "GR", "RHOB", "NPHI", "PEF"]
    buf = parse_columnar(data, num_columns=5, column_names=names, wrapped=True)

    assert buf.num_rows == 2
    assert buf.num_columns == 5
    assert buf["DEPT"] == [1000.0, 1000.5]
    assert buf["GR"] == [25.5, 26.0]
    assert buf["RHOB"] == [120.3, 121.0]
    assert buf["NPHI"] == [0.45, 0.46]
    assert buf["PEF"] == [2.65, 2.66]


def test_ragged_row_detected_in_strict_unwrapped() -> None:
    data = b" 1000.0 25.5 120.3\n 1000.5 26.0\n 1001.0 27.0 122.0\n"
    with pytest.raises(LASDataError, match="column count mismatch"):
        parse_columnar(data, num_columns=3, strict=True)


def test_lenient_mode_pads_missing_columns() -> None:
    data = b" 1000.0 25.5 120.3\n 1000.5 26.0\n 1001.0 27.0 122.0\n"
    buf = parse_columnar(data, num_columns=3, strict=False)
    assert buf.num_rows == 3
    assert buf.column(0) == [1000.0, 1000.5, 1001.0]
    assert buf.column(1) == [25.5, 26.0, 27.0]
    assert buf.column(2)[0] == 120.3
    assert math.isnan(buf.column(2)[1])
    assert buf.column(2)[2] == 122.0


def test_incomplete_row_at_eof() -> None:
    data = b" 1000.0 25.5 120.3\n 1000.5 26.0"
    with pytest.raises(LASDataError):
        parse_columnar(data, num_columns=3, strict=True)


def test_comments_and_blank_lines() -> None:
    data = b"# Header remarks\n\n 100.0 20.0\n# Depth gap\n\n 100.5 21.0\r\n"
    buf = parse_columnar(data, num_columns=2)
    assert buf.num_rows == 2
    assert buf.column(0) == [100.0, 100.5]
    assert buf.column(1) == [20.0, 21.0]


def test_mixed_newlines() -> None:
    data = b" 1.0 2.0\r\n 3.0 4.0\n 5.0 6.0\r"
    buf = parse_columnar(data, num_columns=2)
    assert buf.num_rows == 3
    assert buf.column(0) == [1.0, 3.0, 5.0]
    assert buf.column(1) == [2.0, 4.0, 6.0]


def test_trailing_section_terminates_ascii() -> None:
    data = b" 1000.0 25.0\n 1000.5 26.0\n~Other Remarks\nTrailing non-numeric remarks"
    buf = parse_columnar(data, num_columns=2)
    assert buf.num_rows == 2
    assert buf.column(0) == [1000.0, 1000.5]


def test_empty_data() -> None:
    data = b"# Comment only\n\n"
    with pytest.raises(LASDataError, match="contains no numeric values"):
        parse_columnar(data, num_columns=2, strict=True)

    buf = parse_columnar(data, num_columns=2, strict=False)
    assert buf.num_rows == 0
    assert buf.is_empty


def test_allow_commas() -> None:
    data = b" 100.0, 20.0, 30.0\n 101.0, 21.0, 31.0\n"
    buf = parse_columnar(data, num_columns=3, allow_commas=True)
    assert buf.num_rows == 2
    assert buf.column(0) == [100.0, 101.0]
    assert buf.column(1) == [20.0, 21.0]
    assert buf.column(2) == [30.0, 31.0]


def test_to_dict() -> None:
    data = b" 10.0 20.0\n 30.0 40.0\n"
    buf = parse_columnar(data, num_columns=2, column_names=["X", "Y"])
    d = buf.to_dict()
    assert d == {"X": [10.0, 30.0], "Y": [20.0, 40.0]}


def test_flat_exports() -> None:
    data = b" 1.0 2.0\n 3.0 4.0\n"
    buf = parse_columnar(data, num_columns=2)
    assert buf.to_flat_column_major() == [1.0, 3.0, 2.0, 4.0]
    assert buf.to_flat_row_major() == [1.0, 2.0, 3.0, 4.0]

    flat_bytes = buf.to_flat_bytes(column_major=True)
    assert len(flat_bytes) == 4 * 8
    unpacked = struct.unpack("<4d", flat_bytes)
    assert list(unpacked) == [1.0, 3.0, 2.0, 4.0]


def test_column_statistics() -> None:
    data = b" 10.0 1.0\n 20.0 NaN\n 30.0 3.0\n"
    buf = parse_columnar(data, num_columns=2)

    assert buf.min_max(0) == (10.0, 30.0)
    assert buf.mean(0) == 20.0
    assert buf.null_count(1) == 1
    assert buf.min_max(1) == (1.0, 3.0)
    assert buf.mean(1) == 2.0


def test_column_as_bytes_and_ptr() -> None:
    data = b" 100.0 200.0\n 101.0 201.0\n"
    buf = parse_columnar(data, num_columns=2, column_names=["A", "B"])

    b0 = buf.column_as_bytes(0)
    assert len(b0) == 2 * 8
    vals = struct.unpack("<2d", b0)
    assert list(vals) == [100.0, 101.0]

    b1 = buf.column_as_bytes("B")
    assert len(b1) == 2 * 8
    vals1 = struct.unpack("<2d", b1)
    assert list(vals1) == [200.0, 201.0]

    ptr0 = buf.column_ptr(0)
    assert isinstance(ptr0, int)
    assert ptr0 > 0


def test_column_array_interface() -> None:
    data = b" 10.0 20.0\n 30.0 40.0\n"
    buf = parse_columnar(data, num_columns=2)
    ai = buf.column_array_interface(0)
    assert ai["shape"] == (2,)
    assert ai["typestr"] == "<f8"
    assert ai["version"] == 3
    assert isinstance(ai["data"][0], int)
    assert ai["data"][1] is True


def test_invalid_float_error() -> None:
    data = b" 100.0 BAD_TOKEN 300.0\n"
    with pytest.raises(LASDataError, match="BAD_TOKEN"):
        parse_columnar(data, num_columns=3, column_names=["DEPT", "GR", "RHOB"])


def test_index_and_name_errors() -> None:
    data = b" 10.0 20.0\n"
    buf = parse_columnar(data, num_columns=2, column_names=["A", "B"])

    with pytest.raises(LASDataError, match="out of range"):
        buf.column(5)

    with pytest.raises(LASDataError, match="not found"):
        buf.column("NON_EXISTENT")

    with pytest.raises(LASDataError, match="out of range"):
        buf.row(10)


def test_repr_and_str() -> None:
    data = b" 10.0 20.0\n"
    buf = parse_columnar(data, num_columns=2, column_names=["DEPT", "GR"])
    r = repr(buf)
    assert "rows=1" in r
    assert "cols=2" in r
    assert "DEPT" in r


def test_parser_class_and_properties() -> None:
    parser = ColumnarParser(
        num_columns=3,
        column_names=["A", "B", "C"],
        wrapped=True,
        strict=False,
        capacity_hint=1000,
        allow_commas=True,
    )
    assert parser.num_columns == 3
    assert parser.column_names == ["A", "B", "C"]
    assert parser.is_wrapped
    assert not parser.is_strict
    assert parser.estimate_capacity(500) == 1000
    assert "ColumnarParser" in repr(parser)
