"""Unit tests for LasWell Python mapping interface and zero-copy curve lookups."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
from turbolas import (
    LasCurve,
    LasWell,
    read_las,
)

SAMPLE_LAS_TEXT = """\
~Version Information
VERS. 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP. NO  : ONE LINE PER RECORD FLAG
~Well Information
STRT.M 1000.0 : START DEPTH
STOP.M 1001.0 : STOP DEPTH
STEP.M 0.5    : STEP
NULL. -999.25 : NULL VALUE
WELL. DRILL HOLE #1 : WELL NAME
COMP. ENERGY CORP : COMPANY
FLD . WILDCAT : FIELD
LOC . SEC 12 TWP 34 RGE 5 : LOCATION
SRVC. WIRELINE SERVICES INC : SERVICE COMPANY
DATE. 2026-11-07 : LOG DATE
UWI . 100/01-01-001-01W4/0 : UNIQUE WELL ID
~Curve Information
DEPT.M    : 1 DEPTH
GR  .GAPI : 2 GAMMA RAY
RHOB.K/M3 : 3 BULK DENSITY
~Parameter Information
BHT .DEGC 65.0 : BOTTOM HOLE TEMPERATURE
~Other
This is a test well log dataset.
~Ascii
 1000.0  45.2  2.45
 1000.5  48.7  2.46
 1001.0 -999.25 2.48
"""

WRAPPED_LAS_TEXT = """\
~Version Information
VERS. 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP. YES : ONE LINE PER RECORD FLAG
~Well Information
NULL. -999.25 : NULL VALUE
WELL. WRAPPED HOLE #2 : WELL NAME
~Curve Information
DEPT.M : 1 DEPTH
GR.GAPI : 2 GAMMA RAY
RHOB.K/M3 : 3 BULK DENSITY
~Ascii
 1000.0 45.2
 2.45
 1000.5 48.7
 2.46
"""


@pytest.fixture
def sample_well() -> LasWell:
    return read_las(SAMPLE_LAS_TEXT)


def test_basic_well_properties(sample_well: LasWell) -> None:
    assert sample_well.num_curves == 3
    assert sample_well.num_columns == 3
    assert sample_well.num_rows == 3
    assert sample_well.shape == (3, 3)
    assert not sample_well.is_empty
    assert sample_well.curve_names == ["DEPT", "GR", "RHOB"]
    assert sample_well.keys() == ["DEPT", "GR", "RHOB"]
    assert len(sample_well) == 3


def test_well_metadata_accessors(sample_well: LasWell) -> None:
    assert sample_well.uwi == "100/01-01-001-01W4/0"
    assert sample_well.well_name == "DRILL HOLE #1"
    assert sample_well.company == "ENERGY CORP"
    assert sample_well.field == "WILDCAT"
    assert sample_well.location == "SEC 12 TWP 34 RGE 5"
    assert sample_well.service_company == "WIRELINE SERVICES INC"
    assert sample_well.date == "2026-11-07"
    assert sample_well.start_depth == 1000.0
    assert sample_well.stop_depth == 1001.0
    assert sample_well.step == 0.5
    assert sample_well.null_value == -999.25
    assert sample_well.version == "2.0"
    assert not sample_well.is_wrapped
    assert len(sample_well.version_info) >= 2
    assert len(sample_well.well_info) >= 8
    assert len(sample_well.curve_info) == 3
    assert len(sample_well.parameters) >= 1
    assert len(sample_well.other) >= 1


def test_curve_lookup_numpy_default(sample_well: LasWell) -> None:
    # Default output format is numpy
    assert sample_well.output_format == "numpy"

    gr = sample_well["GR"]
    assert isinstance(gr, np.ndarray)
    assert gr.dtype == np.float64
    assert len(gr) == 3
    assert pytest.approx(gr[0]) == 45.2
    assert pytest.approx(gr[1]) == 48.7
    assert math.isnan(gr[2])  # -999.25 null sentinel mapped to NaN!

    # Zero-copy verification: memory is borrowed, not allocated anew
    assert not gr.flags.owndata
    assert gr.flags.c_contiguous

    dept = sample_well["DEPT"]
    assert np.allclose(dept, [1000.0, 1000.5, 1001.0])


def test_case_insensitive_and_whitespace_lookups(sample_well: LasWell) -> None:
    # Lowercase
    gr_lower = sample_well["gr"]
    assert np.allclose(gr_lower[:2], [45.2, 48.7])

    # Mixed case with whitespace
    gr_spaced = sample_well["  Gr  "]
    assert np.allclose(gr_spaced[:2], [45.2, 48.7])

    # Dot format (MNEM.UNIT fallback)
    dept_dot = sample_well["DEPT.M"]
    assert np.allclose(dept_dot, [1000.0, 1000.5, 1001.0])


def test_integer_indexing(sample_well: LasWell) -> None:
    # 0 -> first curve (DEPT)
    col0 = sample_well[0]
    assert np.allclose(col0, [1000.0, 1000.5, 1001.0])

    # 1 -> second curve (GR)
    col1 = sample_well[1]
    assert np.allclose(col1[:2], [45.2, 48.7])

    # -1 -> last curve (RHOB)
    col_last = sample_well[-1]
    assert np.allclose(col_last, [2.45, 2.46, 2.48])

    # -3 -> DEPT
    assert np.allclose(sample_well[-3], [1000.0, 1000.5, 1001.0])


def test_key_error_on_missing_curve(sample_well: LasWell) -> None:
    with pytest.raises(KeyError, match="CALI.*not found"):
        sample_well["CALI"]

    with pytest.raises(KeyError, match="empty"):
        sample_well["   "]


def test_index_error_out_of_range(sample_well: LasWell) -> None:
    with pytest.raises(IndexError, match="out of range"):
        sample_well[3]

    with pytest.raises(IndexError, match="out of range"):
        sample_well[-4]


def test_type_error_invalid_key_type(sample_well: LasWell) -> None:
    with pytest.raises(TypeError, match="mnemonic or integer index"):
        sample_well[3.14]  # type: ignore[index]


def test_mapping_protocol(sample_well: LasWell) -> None:
    # __contains__
    assert "GR" in sample_well
    assert "gr" in sample_well
    assert "  DEPT  " in sample_well
    assert 0 in sample_well
    assert -1 in sample_well
    assert "UNKNOWN" not in sample_well
    assert 99 not in sample_well
    assert -99 not in sample_well

    # __iter__
    curve_names_iter = list(sample_well)
    assert curve_names_iter == ["DEPT", "GR", "RHOB"]

    # keys(), values(), items()
    assert sample_well.keys() == ["DEPT", "GR", "RHOB"]
    values = sample_well.values()
    assert len(values) == 3
    assert np.allclose(values[0], [1000.0, 1000.5, 1001.0])

    items = sample_well.items()
    assert len(items) == 3
    assert items[0][0] == "DEPT"
    assert np.allclose(items[0][1], [1000.0, 1000.5, 1001.0])

    # get()
    assert np.allclose(sample_well.get("DEPT"), [1000.0, 1000.5, 1001.0])
    assert sample_well.get("MISSING") is None
    assert sample_well.get("MISSING", 999.0) == 999.0


def test_bytes_and_list_output_formats(sample_well: LasWell) -> None:
    sample_well.output_format = "bytes"
    b = sample_well["DEPT"]
    assert isinstance(b, bytes)
    assert len(b) == 3 * 8

    sample_well.output_format = "list"
    lst = sample_well["DEPT"]
    assert isinstance(lst, list)
    assert lst == [1000.0, 1000.5, 1001.0]

    # Reset to numpy
    sample_well.output_format = "numpy"


def test_explicit_curve_accessors(sample_well: LasWell) -> None:
    arr_np = sample_well.curve_numpy("GR")
    assert isinstance(arr_np, np.ndarray)

    mv = sample_well.curve_memoryview("GR")
    assert isinstance(mv, memoryview)
    assert mv.nbytes == 3 * 8

    raw_b = sample_well.curve_bytes("GR")
    assert isinstance(raw_b, bytes)
    assert len(raw_b) == 3 * 8

    lst = sample_well.curve_list("GR")
    assert isinstance(lst, list)
    assert len(lst) == 3


def test_rich_las_curve_object(sample_well: LasWell) -> None:
    curve: LasCurve = sample_well.curve("GR")
    assert curve.mnemonic == "GR"
    assert curve.unit == "GAPI"
    assert "GAMMA RAY" in curve.description
    assert curve.column_index == 1
    assert curve.num_rows == 3
    assert curve.shape == (3,)

    assert curve.null_count() == 1
    assert curve.min_max() == (45.2, 48.7)
    assert pytest.approx(curve.mean() or 0.0) == (45.2 + 48.7) / 2.0

    # __len__ and __getitem__
    assert len(curve) == 3
    assert curve[0] == 45.2
    assert curve[1] == 48.7

    # __array_interface__ zero-copy conversion
    np_arr = np.asarray(curve)
    assert isinstance(np_arr, np.ndarray)
    assert pytest.approx(np_arr[0]) == 45.2

    # as_memoryview
    mv = curve.as_memoryview()
    assert isinstance(mv, memoryview)
    assert mv.nbytes == 3 * 8

    # to_list
    lst = curve.to_list()
    assert lst[:2] == [45.2, 48.7]
    assert math.isnan(lst[2])

    # curves property returning dict
    curves_dict = sample_well.curves
    assert "DEPT" in curves_dict
    assert "GR" in curves_dict
    assert "RHOB" in curves_dict
    assert curves_dict["RHOB"].unit == "K/M3"


def test_whole_table_exports(sample_well: LasWell) -> None:
    # 2D NumPy array
    matrix = sample_well.to_numpy()
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)

    # NumPy dict of 1D arrays
    arr_dict = sample_well.to_numpy(as_dict=True)
    assert isinstance(arr_dict, dict)
    assert "GR" in arr_dict
    assert isinstance(arr_dict["GR"], np.ndarray)

    # Python dict of lists
    py_dict = sample_well.to_dict()
    assert isinstance(py_dict, dict)
    assert py_dict["DEPT"] == [1000.0, 1000.5, 1001.0]

    # Python list of column lists
    col_lists = sample_well.to_list()
    assert isinstance(col_lists, list)
    assert len(col_lists) == 3
    assert col_lists[0] == [1000.0, 1000.5, 1001.0]

    # Optional DataFrame libraries if installed
    try:
        import polars as pl  # type: ignore[import-not-found,import-untyped]

        df_pl = pl.DataFrame(sample_well.to_dict())
        assert isinstance(df_pl, pl.DataFrame)
        assert df_pl.shape == (3, 3)
    except ImportError:
        pass

    try:
        import pandas as pd  # type: ignore[import-untyped]

        df_pd = pd.DataFrame(sample_well.to_dict())
        assert isinstance(df_pd, pd.DataFrame)
        assert df_pd.shape == (3, 3)
    except ImportError:
        pass


def test_wrapped_las_parity() -> None:
    wrapped_well = read_las(WRAPPED_LAS_TEXT)
    assert wrapped_well.is_wrapped
    assert wrapped_well.num_curves == 3
    assert wrapped_well.num_rows == 2
    assert wrapped_well.well_name == "WRAPPED HOLE #2"

    dept = wrapped_well["DEPT"]
    assert np.allclose(dept, [1000.0, 1000.5])

    gr = wrapped_well["GR"]
    assert np.allclose(gr, [45.2, 48.7])

    rhob = wrapped_well["RHOB"]
    assert np.allclose(rhob, [2.45, 2.46])


def test_read_las_from_file_and_stream(tmp_path: Path) -> None:
    # From filesystem Path
    file_path = tmp_path / "test_well.las"
    file_path.write_text(SAMPLE_LAS_TEXT, encoding="utf-8")

    # Read from Path object
    well_path = read_las(file_path)
    assert well_path.num_rows == 3
    assert well_path["GR"][0] == 45.2

    # Read from path string
    well_str = read_las(str(file_path))
    assert well_str.num_rows == 3

    # Read from raw bytes
    well_bytes = read_las(file_path.read_bytes())
    assert well_bytes.num_rows == 3

    # Read from StringIO file-like object
    sio = io.StringIO(SAMPLE_LAS_TEXT)
    well_sio = read_las(sio)
    assert well_sio.num_rows == 3

    # Read from BytesIO file-like object
    bio = io.BytesIO(SAMPLE_LAS_TEXT.encode("utf-8"))
    well_bio = read_las(bio)
    assert well_bio.num_rows == 3


def test_las_well_direct_instantiation(sample_well: LasWell) -> None:
    # Construct LasWell from an existing ColumnarBuffer
    buf = sample_well.data_buffer
    direct_well = LasWell(buf)
    assert direct_well.num_curves == 3
    assert direct_well.num_rows == 3
    assert "GR" in direct_well
    assert np.allclose(direct_well["DEPT"], [1000.0, 1000.5, 1001.0])


def test_header_item_lookup(sample_well: LasWell) -> None:
    from turbolas import SectionType

    item = sample_well.header_item(SectionType.Well, "WELL")
    assert item is not None
    assert item.mnemonic == "WELL"
    assert item.value == "DRILL HOLE #1"

    non_item = sample_well.header_item(SectionType.Well, "NONEXISTENT")
    assert non_item is None


def test_repr_and_str(sample_well: LasWell) -> None:
    r = repr(sample_well)
    assert "LasWell" in r
    assert "100/01-01-001-01W4/0" in r
    assert "DEPT" in r
    assert "rows=3" in r
    assert str(sample_well) == r

    gr_curve = sample_well.curve("GR")
    assert "LasCurve" in repr(gr_curve)
    assert "GAPI" in repr(gr_curve)
