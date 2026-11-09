"""Comprehensive test suite for CWLS LAS 3.0 parsing in TurboLAS.

Validates compliance with CWLS LAS Version 3.0 specification:
- Version 3.0 detection and `is_las3` flag
- Delimiter specification via `DLM . COMMA / SPACE / TAB`
- Compound section naming (`~Log_Definition`, `~Log_Data`, `~Core_Definition`, etc.)
- Multi-section querying (`section_names`, `has_section`, `section_data`, `section_curves`)
- Null sentinel IEEE 754 quiet NaN mapping across distinct datasets
- Zero-copy Python memoryview and native list/dict exports
- Third-party zero-dependency DataFrame construction (Polars, Pandas)
- Strict mode state machine rules (relaxation of LAS 2.0 while enforcing LAS 3.0 constraints)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from turbolas import (
    LASSectionError,
    LasWell,
    ValidationMode,
    read_las,
)

DATA_DIR = Path(__file__).parent / "data"
LAS3_CANONICAL = DATA_DIR / "cwls_30_canonical.las"


@pytest.fixture(scope="module")
def las3_well() -> LasWell:
    return read_las(LAS3_CANONICAL)


class TestCwls30Canonical:
    """Tests targeting canonical CWLS LAS 3.0 multi-section datasets."""

    def test_version_and_flags(self, las3_well: LasWell) -> None:
        assert las3_well.version == "3.0"
        assert las3_well.is_las3 is True
        assert not las3_well.is_wrapped
        assert las3_well.delimiter == "COMMA"

    def test_well_metadata(self, las3_well: LasWell) -> None:
        assert las3_well.well_name == "CANONICAL WELL #3"
        assert las3_well.company == "TURBOLAS ENERGY CORP"
        assert las3_well.field == "NORTH SEA"
        assert las3_well.location == "BLOCK 34/10-A"
        assert las3_well.service_company == "SCHLUMBERGER"
        assert las3_well.date == "2026-11-08"
        assert las3_well.uwi == "100/01-01-001-01W4/0"
        assert las3_well.start_depth == 1670.0
        assert las3_well.stop_depth == 1672.0
        assert las3_well.step == 0.5
        assert las3_well.null_value == -999.25

    def test_section_discovery(self, las3_well: LasWell) -> None:
        sections = las3_well.section_names
        assert "LOG" in sections
        assert "CORE" in sections
        assert "INCLINOMETRY" in sections

        assert las3_well.has_section("LOG")
        assert las3_well.has_section("CORE")
        assert las3_well.has_section("core")  # case-insensitive
        assert las3_well.has_section("INCLINOMETRY")
        assert las3_well.has_section("inclinometry")
        assert not las3_well.has_section("UNKNOWN_SECTION")

    def test_primary_log_dataset(self, las3_well: LasWell) -> None:
        assert las3_well.num_rows == 5
        assert las3_well.num_curves == 4
        assert las3_well.curve_names == ["DEPT", "GR", "NPHI", "RHOB"]

        # First row values
        assert pytest.approx(las3_well["DEPT"][0]) == 1670.0
        assert pytest.approx(las3_well["GR"][0]) == 45.2
        assert pytest.approx(las3_well["NPHI"][0]) == 0.215
        assert pytest.approx(las3_well["RHOB"][0]) == 2.45

        # Null sentinel mapping
        gr_col = las3_well["GR"]
        assert math.isnan(gr_col[2])
        rhob_col = las3_well["RHOB"]
        assert math.isnan(rhob_col[3])
        nphi_col = las3_well["NPHI"]
        assert math.isnan(nphi_col[4])

    def test_core_dataset_access(self, las3_well: LasWell) -> None:
        core_buf = las3_well.section_data("CORE")
        assert core_buf is not None
        assert core_buf.num_rows == 3
        assert core_buf.num_columns == 4
        assert core_buf.column_names == ["CDEP", "CPOR", "CPERM", "CGD"]

        # Core porosity values
        cpor = core_buf.column("CPOR")
        assert cpor == [18.5, 17.8, 19.1]

        # Core permeability values
        cperm = core_buf.column("CPERM")
        assert cperm == [125.4, 98.2, 152.0]

        # Check section curve headers
        core_curves = las3_well.section_curves("CORE")
        assert core_curves is not None
        assert len(core_curves) == 4
        assert core_curves[0].mnemonic == "CDEP"
        assert core_curves[0].unit == "M"
        assert core_curves[1].mnemonic == "CPOR"
        assert core_curves[1].unit == "%"

    def test_inclinometry_dataset_access(self, las3_well: LasWell) -> None:
        inc_buf = las3_well.section_data("INCLINOMETRY")
        assert inc_buf is not None
        assert inc_buf.num_rows == 3
        assert inc_buf.num_columns == 3
        assert inc_buf.column_names == ["DEPT", "HAZI", "DEVI"]

        devi = inc_buf.column("DEVI")
        assert devi == [2.1, 2.3, 2.45]

    def test_sections_dictionary(self, las3_well: LasWell) -> None:
        sec_map = las3_well.sections
        assert isinstance(sec_map, dict)
        assert "LOG" in sec_map
        assert "CORE" in sec_map
        assert "INCLINOMETRY" in sec_map

        assert sec_map["LOG"].num_rows == 5
        assert sec_map["CORE"].num_rows == 3
        assert sec_map["INCLINOMETRY"].num_rows == 3

    def test_native_datatype_exports(self, las3_well: LasWell) -> None:
        # 1. to_dict() returns dict[str, list[float]]
        d = las3_well.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {"DEPT", "GR", "NPHI", "RHOB"}
        assert len(d["DEPT"]) == 5
        assert d["DEPT"] == [1670.0, 1670.5, 1671.0, 1671.5, 1672.0]

        # 2. to_list() returns list[list[float]]
        cols = las3_well.to_list()
        assert isinstance(cols, list)
        assert len(cols) == 4
        assert cols[0] == [1670.0, 1670.5, 1671.0, 1671.5, 1672.0]

        # 3. curve_memoryview() returns zero-copy Python memoryview
        mv = las3_well.curve_memoryview("DEPT")
        assert isinstance(mv, memoryview)
        assert mv.nbytes == 5 * 8
        assert mv.format == "B"

        # 4. curve_bytes() returns raw bytes
        b = las3_well.curve_bytes("DEPT")
        assert isinstance(b, bytes)
        assert len(b) == 5 * 8

        # 5. to_numpy() returns 2D ndarray
        matrix = las3_well.to_numpy()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (5, 4)

    def test_downstream_dataframe_ingestion(self, las3_well: LasWell) -> None:
        """Verifies downstream libraries consume native python dict without extra deps."""
        data_dict = las3_well.to_dict()

        try:
            import polars as pl

            df_pl = pl.DataFrame(data_dict)
            assert isinstance(df_pl, pl.DataFrame)
            assert df_pl.shape == (5, 4)
            assert df_pl.columns == ["DEPT", "GR", "NPHI", "RHOB"]
            assert df_pl["DEPT"][0] == 1670.0
        except ImportError:
            pass

        try:
            import pandas as pd

            df_pd = pd.DataFrame(data_dict)
            assert isinstance(df_pd, pd.DataFrame)
            assert df_pd.shape == (5, 4)
            assert list(df_pd.columns) == ["DEPT", "GR", "NPHI", "RHOB"]
            assert df_pd["DEPT"].iloc[0] == 1670.0
        except ImportError:
            pass

    def test_strict_mode_compliance(self) -> None:
        """In strict mode, LAS 3.0 allows multiple data sections and sections after data."""
        well = read_las(LAS3_CANONICAL, validation_mode=ValidationMode.Strict)
        assert well.is_las3 is True
        assert well.num_rows == 5
        assert well.has_section("CORE")

    def test_las3_space_delimiter(self) -> None:
        """Tests LAS 3.0 file with DLM . SPACE."""
        las3_space = """\
~Version Information
VERS. 3.0 : CWLS
WRAP. NO  : NO
DLM . SPACE : DELIMITER
~Well Information
WELL. SPACE WELL :
NULL. -999.25 :
~Log_Definition
DEPT.M :
GR.GAPI :
~Log_Data
1000.0 45.2
1000.5 48.7
~Core_Definition
CDEP.M :
CPOR.% :
~Core_Data
1000.2 18.5
"""
        well = read_las(las3_space, validation_mode=ValidationMode.Strict)
        assert well.is_las3 is True
        assert well.delimiter == "SPACE"
        assert well.num_rows == 2
        assert well.curve_names == ["DEPT", "GR"]
        assert well["DEPT"] == pytest.approx([1000.0, 1000.5])
        assert well.has_section("CORE")
        core_buf = well.section_data("CORE")
        assert core_buf is not None
        assert core_buf.num_rows == 1
        assert core_buf.column("CPOR") == [18.5]

    def test_las3_tab_delimiter(self) -> None:
        """Tests LAS 3.0 file with DLM . TAB."""
        las3_tab = """\
~Version Information
VERS. 3.0 : CWLS
WRAP. NO  : NO
DLM . TAB : DELIMITER
~Well Information
WELL. TAB WELL :
NULL. -999.25 :
~Log_Definition
DEPT.M :
GR.GAPI :
~Log_Data
1000.0\t45.2
1000.5\t48.7
"""
        well = read_las(las3_tab, validation_mode=ValidationMode.Strict)
        assert well.is_las3 is True
        assert well.delimiter == "TAB"
        assert well.num_rows == 2
        assert well["GR"] == pytest.approx([45.2, 48.7])

    def test_las3_duplicate_version_rejected_in_strict(self) -> None:
        """Duplicate ~Version section in LAS 3.0 must raise an error in strict mode."""
        bad_las3 = """\
~Version Information
VERS. 3.0 :
WRAP. NO :
~Well Information
WELL. TEST :
~Version Information
VERS. 3.0 :
~Log_Definition
DEPT.M :
~Log_Data
1000.0
"""
        with pytest.raises(LASSectionError):
            read_las(bad_las3, validation_mode=ValidationMode.Strict)
