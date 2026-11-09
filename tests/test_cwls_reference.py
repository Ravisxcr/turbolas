"""End-to-end integration test suite targeting canonical CWLS LAS 1.2 and 2.0 reference datasets.

Validates parser compliance against real-world well log edge cases:
- Canonical CWLS 1.2 and 2.0 reference datasets
- Wrapped and unwrapped (`WRAP = YES/NO`) ASCII data sections
- Exact numerical parity between wrapped and unwrapped datasets
- Custom delimiter variants (spaces, tabs, mixed whitespace, CRLF)
- Mixed null sentinels and IEEE 754 quiet NaN mapping
- Zero-copy decoding integrity across UTF-8, BOM, and Windows-1252 (CP1252)
- Channel ordering and native List / NumPy export integrity
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pytest
from turbolas import (
    LasCurve,
    LasWell,
    OutputFormat,
    SectionType,
    ValidationMode,
    read_las,
)

DATA_DIR = Path(__file__).parent / "data"


def is_quiet_nan(val: float) -> bool:
    """Verifies that a float value is an IEEE 754 quiet NaN."""
    if not math.isnan(val):
        return False
    raw_bytes = np.float64(val).tobytes()
    # In little-endian IEEE 754 float64, bit 51 is the quiet/signaling NaN bit.
    bits = int.from_bytes(raw_bytes, byteorder="little")
    return (bits & 0x0008_0000_0000_0000) != 0


# ===========================================================================
# Module-Scoped Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def cwls_12_well() -> LasWell:
    return read_las(DATA_DIR / "cwls_12_canonical.las")


@pytest.fixture(scope="module")
def cwls_20_unwrapped_well() -> LasWell:
    return read_las(DATA_DIR / "cwls_20_canonical_unwrapped.las")


@pytest.fixture(scope="module")
def cwls_20_wrapped_well() -> LasWell:
    return read_las(DATA_DIR / "cwls_20_canonical_wrapped.las")


@pytest.fixture(scope="module")
def cwls_20_custom_sentinel_well() -> LasWell:
    return read_las(DATA_DIR / "cwls_20_custom_sentinel.las")


@pytest.fixture(scope="module")
def cwls_20_tab_delimited_well() -> LasWell:
    return read_las(DATA_DIR / "cwls_20_tab_delimited_crlf.las")


@pytest.fixture(scope="module")
def cwls_20_cp1252_well() -> LasWell:
    return read_las(DATA_DIR / "cwls_20_cp1252_encoding.las")


@pytest.fixture(scope="module")
def cwls_20_diverse_headers_well() -> LasWell:
    return read_las(DATA_DIR / "cwls_20_diverse_headers.las")


# ===========================================================================
# 1. Canonical CWLS LAS 1.2 Reference Dataset
# ===========================================================================


class TestCwls12Canonical:
    """Tests targeting canonical CWLS LAS 1.2 reference dataset."""

    def test_version_and_wrap(self, cwls_12_well: LasWell) -> None:
        assert cwls_12_well.version == "1.2"
        assert not cwls_12_well.is_wrapped

    def test_well_metadata(self, cwls_12_well: LasWell) -> None:
        assert cwls_12_well.uwi == "100123401234W400"
        assert cwls_12_well.well_name == "ANY ET AL 12-34-12-34"
        assert cwls_12_well.company == "ANY OIL COMPANY INC."
        assert cwls_12_well.field == "EDAM"
        assert cwls_12_well.location == "A12-34-12-34W4M"
        assert cwls_12_well.service_company == "ANY LOGGING COMPANY INC."
        assert cwls_12_well.date == "25-DEC-1988"
        assert cwls_12_well.start_depth == 1670.0
        assert cwls_12_well.stop_depth == 1660.0
        assert cwls_12_well.step == -0.125
        assert cwls_12_well.null_value == -999.25

    def test_channel_order_and_counts(self, cwls_12_well: LasWell) -> None:
        expected = ["DEPT", "DT", "RHOB", "NPHI", "SFLU", "SFLA", "ILM", "ILD"]
        assert cwls_12_well.curve_names == expected
        assert cwls_12_well.num_curves == 8
        assert cwls_12_well.num_rows == 5
        assert len(cwls_12_well) == 8

    def test_curve_metadata_extractions(self, cwls_12_well: LasWell) -> None:
        dt: LasCurve = cwls_12_well.curve("DT")
        assert dt.mnemonic == "DT"
        assert dt.unit == "US/M"
        assert dt.value == "60 520 01 00"
        assert "SONIC TRANSIT TIME" in dt.description

    def test_parameter_extractions(self, cwls_12_well: LasWell) -> None:
        bht = cwls_12_well.header_item(SectionType.Parameter, "BHT")
        assert bht is not None
        assert bht.value == "35.5000"
        assert bht.unit == "DEGC"

        bs = cwls_12_well.header_item(SectionType.Parameter, "BS")
        assert bs is not None
        assert bs.value == "200.0000"

    def test_other_remarks(self, cwls_12_well: LasWell) -> None:
        assert len(cwls_12_well.other) >= 1
        assert "stuck at 1665 meters" in cwls_12_well.other[0]

    def test_exact_numerical_curve_parity(self, cwls_12_well: LasWell) -> None:
        dept = cwls_12_well["DEPT"]
        np.testing.assert_allclose(dept, [1670.0, 1669.875, 1669.750, 1669.625, 1669.500])

        dt = cwls_12_well["DT"]
        assert pytest.approx(dt[0]) == 123.45
        assert pytest.approx(dt[1]) == 123.45
        assert pytest.approx(dt[2]) == 124.10
        assert np.isnan(dt[3])
        assert is_quiet_nan(dt[3])
        assert np.isnan(dt[4])
        assert is_quiet_nan(dt[4])

        rhob = cwls_12_well["RHOB"]
        assert pytest.approx(rhob[0]) == 2550.0
        assert pytest.approx(rhob[1]) == 2550.0
        assert pytest.approx(rhob[2]) == 2548.5
        assert pytest.approx(rhob[3]) == 2545.0
        assert np.isnan(rhob[4])
        assert is_quiet_nan(rhob[4])

        ilm = cwls_12_well["ILM"]
        np.testing.assert_allclose(ilm, [110.2, 110.2, 111.0, 112.5, 113.0])

        ild = cwls_12_well["ILD"]
        np.testing.assert_allclose(ild, [105.6, 105.6, 106.2, 107.0, 108.4])

    def test_zero_copy_view_semantics(self, cwls_12_well: LasWell) -> None:
        dept = cwls_12_well["DEPT"]
        assert not dept.flags.owndata
        assert dept.flags.c_contiguous
        assert dept.dtype == np.float64


# ===========================================================================
# 2. Canonical CWLS LAS 2.0 Unwrapped Reference Dataset
# ===========================================================================


class TestCwls20UnwrappedCanonical:
    """Tests targeting canonical CWLS LAS 2.0 unwrapped reference dataset."""

    def test_version_and_wrap(self, cwls_20_unwrapped_well: LasWell) -> None:
        assert cwls_20_unwrapped_well.version == "2.0"
        assert not cwls_20_unwrapped_well.is_wrapped

    def test_dimensions_and_channels(self, cwls_20_unwrapped_well: LasWell) -> None:
        assert cwls_20_unwrapped_well.num_curves == 8
        assert cwls_20_unwrapped_well.num_rows == 6
        expected = ["DEPT", "DT", "RHOB", "NPHI", "SFLU", "SFLA", "ILM", "ILD"]
        assert cwls_20_unwrapped_well.curve_names == expected

    def test_depth_channel(self, cwls_20_unwrapped_well: LasWell) -> None:
        dept = cwls_20_unwrapped_well["DEPT"]
        np.testing.assert_allclose(dept, [1670.0, 1669.875, 1669.750, 1669.625, 1669.500, 1669.375])

    def test_null_sentinel_mapping(self, cwls_20_unwrapped_well: LasWell) -> None:
        dt = cwls_20_unwrapped_well["DT"]
        assert pytest.approx(dt[0]) == 123.45
        assert pytest.approx(dt[4]) == 126.25
        # Row 5: -999.2500 null sentinel mapped to quiet NaN
        assert np.isnan(dt[5])
        assert is_quiet_nan(dt[5])


# ===========================================================================
# 3. Canonical CWLS LAS 2.0 Wrapped Parity
# ===========================================================================


class TestCwls20WrappedParity:
    """Tests asserting exact numerical parity between unwrapped and wrapped modes."""

    def test_wrap_flags(
        self, cwls_20_unwrapped_well: LasWell, cwls_20_wrapped_well: LasWell
    ) -> None:
        assert not cwls_20_unwrapped_well.is_wrapped
        assert cwls_20_wrapped_well.is_wrapped

    def test_dimension_equality(
        self, cwls_20_unwrapped_well: LasWell, cwls_20_wrapped_well: LasWell
    ) -> None:
        assert cwls_20_unwrapped_well.num_curves == cwls_20_wrapped_well.num_curves == 8
        assert cwls_20_unwrapped_well.num_rows == cwls_20_wrapped_well.num_rows == 6
        assert cwls_20_unwrapped_well.curve_names == cwls_20_wrapped_well.curve_names

    def test_exact_channel_numerical_parity(
        self, cwls_20_unwrapped_well: LasWell, cwls_20_wrapped_well: LasWell
    ) -> None:
        for name in cwls_20_unwrapped_well.curve_names:
            u_col = cwls_20_unwrapped_well[name]
            w_col = cwls_20_wrapped_well[name]

            # 1. Exact numerical comparison including NaNs
            np.testing.assert_allclose(u_col, w_col, equal_nan=True)

            # 2. NaN mask parity
            np.testing.assert_array_equal(np.isnan(u_col), np.isnan(w_col))

            # 3. Quiet NaN verification on all null positions
            for val in w_col:
                if math.isnan(val):
                    assert is_quiet_nan(val)

            # 4. Bitwise equality on non-NaN entries
            for u_val, w_val in zip(u_col, w_col):
                if not math.isnan(u_val):
                    assert np.float64(u_val).tobytes() == np.float64(w_val).tobytes()


# ===========================================================================
# 4. Custom Null Sentinel and Mixed Sentinels
# ===========================================================================


class TestCustomNullSentinel:
    """Tests dynamic NULL sentinel detection and IEEE 754 quiet NaN mapping."""

    def test_declared_null_value(self, cwls_20_custom_sentinel_well: LasWell) -> None:
        assert cwls_20_custom_sentinel_well.null_value == -9999.0

    def test_custom_sentinel_mapping(self, cwls_20_custom_sentinel_well: LasWell) -> None:
        gr = cwls_20_custom_sentinel_well["GR"]
        assert pytest.approx(gr[0]) == 55.40
        # Row 1: -9999.00 mapped to NaN
        assert np.isnan(gr[1])
        assert is_quiet_nan(gr[1])
        assert pytest.approx(gr[2]) == 58.20
        # Row 3: -999.25 is NOT the declared sentinel (-9999.00), so it is kept as -999.25!
        assert pytest.approx(gr[3]) == -999.25
        assert not np.isnan(gr[3])
        assert pytest.approx(gr[4]) == 62.00

    def test_text_nan_mapping(self, cwls_20_custom_sentinel_well: LasWell) -> None:
        dtc = cwls_20_custom_sentinel_well["DTC"]
        assert pytest.approx(dtc[0]) == 72.40
        # Row 3: text 'NaN' mapped to quiet NaN
        assert np.isnan(dtc[3])
        assert is_quiet_nan(dtc[3])

    def test_curve_statistics_with_custom_null(self, cwls_20_custom_sentinel_well: LasWell) -> None:
        gr_curve = cwls_20_custom_sentinel_well.curve("GR")
        assert gr_curve.null_count() == 1
        min_val, max_val = gr_curve.min_max() or (0.0, 0.0)
        assert pytest.approx(min_val) == -999.25
        assert pytest.approx(max_val) == 62.00


# ===========================================================================
# 5. Tab Delimited and CRLF Log Data
# ===========================================================================


class TestTabDelimitedCrlf:
    """Tests tab-delimited ASCII data, mixed spaces/tabs, CRLF, and comments."""

    def test_structure(self, cwls_20_tab_delimited_well: LasWell) -> None:
        assert cwls_20_tab_delimited_well.curve_names == ["DEPT", "GR", "RHOB", "NPHI"]
        assert cwls_20_tab_delimited_well.num_rows == 5
        assert cwls_20_tab_delimited_well.start_depth == 2000.0
        assert cwls_20_tab_delimited_well.stop_depth == 2002.0

    def test_curve_values(self, cwls_20_tab_delimited_well: LasWell) -> None:
        dept = cwls_20_tab_delimited_well["DEPT"]
        np.testing.assert_allclose(dept, [2000.0, 2000.5, 2001.0, 2001.5, 2002.0])

        gr = cwls_20_tab_delimited_well["GR"]
        assert pytest.approx(gr[0]) == 45.5
        assert pytest.approx(gr[1]) == 48.2
        assert np.isnan(gr[2])  # -999.25
        assert is_quiet_nan(gr[2])
        assert pytest.approx(gr[3]) == 52.1
        assert pytest.approx(gr[4]) == 55.0


# ===========================================================================
# 6. Windows-1252 (CP1252) Encoding Fallback
# ===========================================================================


class TestCp1252Encoding:
    """Tests SIMD UTF-8 validation fallback to Windows-1252 directly against byte buffers."""

    def test_metadata_special_characters(self, cwls_20_cp1252_well: LasWell) -> None:
        assert cwls_20_cp1252_well.well_name == "“TROLL FIELD EXTENSION °1”"
        assert cwls_20_cp1252_well.company == "“TOTAL ENERGIES EP NORGE AS”"

    def test_units_with_micro_and_degree(self, cwls_20_cp1252_well: LasWell) -> None:
        dt = cwls_20_cp1252_well.curve("DT")
        assert dt.unit == "µS/M"

        temp = cwls_20_cp1252_well.curve("TEMP")
        assert temp.unit == "°C"

    def test_parameter_special_characters(self, cwls_20_cp1252_well: LasWell) -> None:
        bht = cwls_20_cp1252_well.header_item(SectionType.Parameter, "BHT")
        assert bht is not None
        assert bht.unit == "°C"
        assert bht.value == "78.5000"

    def test_remarks_with_copyright_and_euro(self, cwls_20_cp1252_well: LasWell) -> None:
        assert len(cwls_20_cp1252_well.other) >= 1
        assert "©" in cwls_20_cp1252_well.other[0]
        assert "€" in cwls_20_cp1252_well.other[0]

    def test_numeric_data_integrity(self, cwls_20_cp1252_well: LasWell) -> None:
        temp = cwls_20_cp1252_well["TEMP"]
        np.testing.assert_allclose(temp, [68.5, 69.0, 69.5, 70.0])


# ===========================================================================
# 7. Diverse Header Configurations and Version Quirks
# ===========================================================================


class TestDiverseHeaders:
    """Tests case-insensitivity of section delimiters, embedded colons, and quotes."""

    def test_version_and_well(self, cwls_20_diverse_headers_well: LasWell) -> None:
        assert cwls_20_diverse_headers_well.version == "2.0"
        assert cwls_20_diverse_headers_well.well_name == "DIVERSE RIG #4"
        assert cwls_20_diverse_headers_well.company == '"CHEVRON CANADA RESOURCES"'

    def test_parameters_with_colons(self, cwls_20_diverse_headers_well: LasWell) -> None:
        time_item = cwls_20_diverse_headers_well.header_item(SectionType.Parameter, "TIME")
        assert time_item is not None
        assert time_item.value == "14:35:00"

        scale_item = cwls_20_diverse_headers_well.header_item(SectionType.Parameter, "SCALE")
        assert scale_item is not None
        assert scale_item.value == "1:240"

        run_item = cwls_20_diverse_headers_well.header_item(SectionType.Parameter, "RUN")
        assert run_item is not None
        assert run_item.value == "2"
        assert run_item.description == "MAIN PASS : FROM 5000 TO 5003 FT"

    def test_missing_optional_sections(self, cwls_20_diverse_headers_well: LasWell) -> None:
        assert cwls_20_diverse_headers_well.other == []

    def test_curve_data(self, cwls_20_diverse_headers_well: LasWell) -> None:
        gr = cwls_20_diverse_headers_well["GR"]
        assert pytest.approx(gr[0]) == 62.5
        assert pytest.approx(gr[1]) == 65.1
        assert pytest.approx(gr[2]) == 68.4
        assert np.isnan(gr[3])


# ===========================================================================
# 8. Export Paths & Ingestion Sources
# ===========================================================================


class TestExportsAndIngestionSources:
    """Tests whole-table native List / NumPy exports and diverse ingestion sources."""

    def test_to_list_export(self, cwls_12_well: LasWell) -> None:
        cols = cwls_12_well.to_list()
        assert isinstance(cols, list)
        assert len(cols) == 8
        assert len(cols[0]) == 5
        assert pytest.approx(cols[0][0]) == 1670.0

    def test_numpy_2d_export(self, cwls_12_well: LasWell) -> None:
        matrix = cwls_12_well.to_numpy()
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (5, 8)
        assert matrix.dtype == np.float64
        # Verify first row
        assert pytest.approx(matrix[0, 0]) == 1670.0
        assert pytest.approx(matrix[0, 1]) == 123.45

    def test_to_dict_export(self, cwls_12_well: LasWell) -> None:
        d = cwls_12_well.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 8
        assert len(d["DEPT"]) == 5

    def test_ingest_from_bytes_and_stream(self) -> None:
        path = DATA_DIR / "cwls_12_canonical.las"
        raw_bytes = path.read_bytes()

        # From bytes
        well_from_bytes = read_las(raw_bytes)
        assert well_from_bytes.num_rows == 5

        # From BytesIO
        well_from_bio = read_las(io.BytesIO(raw_bytes))
        assert well_from_bio.num_rows == 5

        # From StringIO
        well_from_sio = read_las(io.StringIO(raw_bytes.decode("utf-8")))
        assert well_from_sio.num_rows == 5

        # From open file
        with open(path, "rb") as f:
            well_from_file = read_las(f)
            assert well_from_file.num_rows == 5

    def test_output_format_list_and_bytes(self) -> None:
        well = read_las(DATA_DIR / "cwls_12_canonical.las", output_format="list")
        assert well.output_format == "list"
        dept_list = well["DEPT"]
        assert isinstance(dept_list, list)
        assert len(dept_list) == 5

        well.output_format = OutputFormat.Bytes
        assert well.output_format == "bytes"
        dept_bytes = well["DEPT"]
        assert isinstance(dept_bytes, bytes)
        assert len(dept_bytes) == 5 * 8

        # Test memoryview export
        mv = well.curve_memoryview("DEPT")
        assert isinstance(mv, memoryview)
        assert mv.nbytes == 5 * 8

    def test_validation_mode_propagation(self) -> None:
        well = read_las(
            DATA_DIR / "cwls_12_canonical.las",
            validation_mode=ValidationMode.Permissive,
        )
        assert well.num_rows == 5
