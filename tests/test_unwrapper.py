"""Unit tests for streaming line-unwrapping engine (CWLS LAS WRAP = YES)."""

from __future__ import annotations

import math

import pytest
from turbolas import (
    ColumnarBuffer,
    LASDataError,
    NullSentinel,
    StreamingUnwrapper,
    UnwrappedRecord,
    decode_buffer,
    parse_columnar,
    unwrap_and_parse,
)


def test_streaming_unwrapper_iteration() -> None:
    # 5 channels folded across 2 lines per depth step
    raw = b" 1000.0  25.5  120.3\n  0.45  2.65\n 1000.5  26.0  121.0\n  0.46  2.66\n"
    names = ["DEPT", "GR", "RHOB", "NPHI", "PEF"]

    unwrapper = StreamingUnwrapper(raw, expected_channels=5, column_names=names)
    assert unwrapper.expected_channels == 5
    assert unwrapper.record_index == 0

    records: list[UnwrappedRecord] = []
    for record in unwrapper:
        assert isinstance(record, UnwrappedRecord)
        records.append(record)

    assert len(records) == 2

    # Record 0
    r0 = records[0]
    assert r0.record_index == 0
    assert r0.start_line == 1
    assert r0.end_line == 2
    assert r0.line_count == 2
    assert len(r0) == 5
    assert r0.values == [1000.0, 25.5, 120.3, 0.45, 2.65]
    assert r0[0] == 1000.0
    assert r0[-1] == 2.65

    # Record 1
    r1 = records[1]
    assert r1.record_index == 1
    assert r1.start_line == 3
    assert r1.end_line == 4
    assert r1.line_count == 2
    assert r1.values == [1000.5, 26.0, 121.0, 0.46, 2.66]


def test_streaming_unwrapper_three_lines_per_depth() -> None:
    # 7 channels folded across 3 lines: 1 + 3 + 3
    raw = (
        b" 1000.0\n 25.5 120.3 0.45\n 2.65 15.0 45.0\n 1000.5\n 26.0 121.0 0.46\n 2.66 16.0 46.0\n"
    )
    unwrapper = StreamingUnwrapper(raw, expected_channels=7)
    records = unwrapper.read_records()
    assert len(records) == 2

    assert records[0].start_line == 1
    assert records[0].end_line == 3
    assert records[0].line_count == 3
    assert records[0].values == [1000.0, 25.5, 120.3, 0.45, 2.65, 15.0, 45.0]

    assert records[1].start_line == 4
    assert records[1].end_line == 6
    assert records[1].line_count == 3
    assert records[1].values == [1000.5, 26.0, 121.0, 0.46, 2.66, 16.0, 46.0]


def test_read_all_returns_columnar_buffer() -> None:
    raw = b" 1000.0 25.0 120.0\n 0.40 2.50\n 1000.5 26.0 121.0\n 0.41 2.51\n"
    names = ["DEPT", "GR", "RHOB", "NPHI", "PEF"]
    unwrapper = StreamingUnwrapper(raw, expected_channels=5, column_names=names)
    buf = unwrapper.read_all()

    assert isinstance(buf, ColumnarBuffer)
    assert buf.num_rows == 2
    assert buf.num_columns == 5
    assert buf.column_names == names
    assert buf["DEPT"] == [1000.0, 1000.5]
    assert buf["PEF"] == [2.50, 2.51]


def test_unwrap_and_parse_convenience_function() -> None:
    raw = b" 1000.0 25.0\n 120.0\n 1000.5 26.0\n 121.0\n"
    names = ["DEPT", "GR", "RHOB"]
    buf = unwrap_and_parse(raw, expected_channels=3, column_names=names)

    assert buf.num_rows == 2
    assert buf.num_columns == 3
    assert buf["DEPT"] == [1000.0, 1000.5]
    assert buf["RHOB"] == [120.0, 121.0]


def test_parity_between_wrapped_and_unwrapped() -> None:
    # 4 channels: DEPT, GR, RHOB, NPHI
    wrapped_raw = b" 1000.0 25.0\n 120.0 0.40\n 1000.5 26.0\n 121.0 0.41\n"
    unwrapped_raw = b" 1000.0 25.0 120.0 0.40\n 1000.5 26.0 121.0 0.41\n"
    names = ["DEPT", "GR", "RHOB", "NPHI"]

    buf_wrapped = parse_columnar(wrapped_raw, num_columns=4, column_names=names, wrapped=True)
    buf_unwrapped = parse_columnar(unwrapped_raw, num_columns=4, column_names=names, wrapped=False)

    assert buf_wrapped.num_rows == buf_unwrapped.num_rows
    assert buf_wrapped.num_columns == buf_unwrapped.num_columns
    assert buf_wrapped.column_names == buf_unwrapped.column_names

    for col in names:
        assert buf_wrapped[col] == buf_unwrapped[col]


def test_null_sentinel_handling() -> None:
    raw = b" 1000.0 -999.25\n null NaN\n"
    sentinel = NullSentinel.default_cwls()
    buf = unwrap_and_parse(raw, expected_channels=4, null_sentinel=sentinel)

    assert buf.num_rows == 1
    assert buf.column(0) == [1000.0]
    assert math.isnan(buf.column(1)[0])
    assert math.isnan(buf.column(2)[0])
    assert math.isnan(buf.column(3)[0])


def test_strict_error_on_truncated_record() -> None:
    # Record 1 complete (4 channels). Record 2 truncated after 2 channels.
    raw = b" 1000.0 25.0\n 120.0 0.40\n 1000.5 26.0"
    with pytest.raises(LASDataError):
        unwrap_and_parse(raw, expected_channels=4, strict=True)


def test_recovery_mode_pads_truncated_record() -> None:
    raw = b" 1000.0 25.0\n 120.0 0.40\n 1000.5 26.0"
    buf = unwrap_and_parse(raw, expected_channels=4, strict=False)
    assert buf.num_rows == 2
    assert buf.column(0) == [1000.0, 1000.5]
    assert buf.column(1) == [25.0, 26.0]
    assert buf.column(2)[0] == 120.0
    assert math.isnan(buf.column(2)[1])
    assert buf.column(3)[0] == 0.40
    assert math.isnan(buf.column(3)[1])


def test_comments_and_blank_lines() -> None:
    raw = b"# Remark\n\n 1000.0 25.0\n# Another remark\n\n 120.0 0.40\r\n"
    buf = unwrap_and_parse(raw, expected_channels=4)
    assert buf.num_rows == 1
    assert buf.column(0) == [1000.0]
    assert buf.column(1) == [25.0]
    assert buf.column(2) == [120.0]
    assert buf.column(3) == [0.40]


def test_trailing_section_terminates_cleanly() -> None:
    raw = b" 1000.0 25.0\n 120.0 0.40\n~Other Information\nNon-numeric remarks..."
    buf = unwrap_and_parse(raw, expected_channels=4)
    assert buf.num_rows == 1
    assert buf.column(0) == [1000.0]


def test_input_types() -> None:
    text = " 1000.0 25.0\n 120.0 0.40\n"
    # Str input
    buf_str = unwrap_and_parse(text, expected_channels=4)
    assert buf_str.num_rows == 1

    # DecodedBuffer input
    decoded = decode_buffer(text.encode("ascii"))
    buf_decoded = unwrap_and_parse(decoded, expected_channels=4)
    assert buf_decoded.num_rows == 1


def test_record_indexing_and_repr() -> None:
    raw = b" 1000.0 25.0\n 120.0 0.40\n"
    unwrapper = StreamingUnwrapper(raw, expected_channels=4)
    rec = unwrapper.next_record()
    assert rec is not None
    assert rec[0] == 1000.0
    assert rec[-1] == 0.40

    with pytest.raises(LASDataError, match="out of range"):
        _ = rec[10]

    assert "index=0" in repr(rec)
    assert "channels=4" in repr(rec)
    assert "StreamingUnwrapper" in repr(unwrapper)
