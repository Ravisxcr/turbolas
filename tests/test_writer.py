"""Comprehensive test suite for TurboLAS Rust-backed LAS writer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import turbolas


def test_write_to_string_and_roundtrip_las2() -> None:
    """Tests serializing a LAS 2.0 well to string and parsing it back."""
    sample_las = (
        "~VERSION INFORMATION\n"
        " VERS. 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0\n"
        " WRAP. NO : ONE LINE PER DEPTH STEP\n"
        "~WELL INFORMATION\n"
        "#MNEM.UNIT              DATA                   DESCRIPTION\n"
        "#---------              ----                   -----------\n"
        " STRT.M                 1000.0000              : START DEPTH\n"
        " STOP.M                 1001.0000              : STOP DEPTH\n"
        " STEP.M                 0.5000                 : STEP VALUE\n"
        " NULL.                  -999.2500              : NULL VALUE\n"
        " COMP.                  TEST CORP              : COMPANY\n"
        " WELL.                  TEST WELL #1           : WELL NAME\n"
        " UWI .                  1234567890             : UNIQUE WELL ID\n"
        "~CURVE INFORMATION\n"
        "#MNEM.UNIT             API CODE                CURVE DESCRIPTION\n"
        "#---------             --------                -----------------\n"
        " DEPT.M                                        : Depth curve\n"
        " GR  .GAPI                                     : Gamma ray\n"
        " NPHI.V/V                                      : Neutron porosity\n"
        "~A\n"
        "  1000.0000     45.2000      0.1500\n"
        "  1000.5000     52.1000     -999.2500\n"
        "  1001.0000     48.9000      0.1450\n"
    )

    well = turbolas.read(sample_las)
    assert well.num_rows == 3
    assert well.num_curves == 3

    # Serialize using well.write() returning string
    output_str = well.write(version=2.0)
    assert isinstance(output_str, str)
    assert "~VERSION INFORMATION" in output_str
    assert "VERS.   2.0" in output_str
    assert "~WELL INFORMATION" in output_str
    assert "~CURVE INFORMATION" in output_str
    assert "~A" in output_str

    # Parse the serialized string back with TurboLAS
    well_rt = turbolas.read(output_str)
    assert well_rt.num_rows == 3
    assert well_rt.num_curves == 3
    assert well_rt.curve_names == ["DEPT", "GR", "NPHI"]
    assert well_rt.well_name == "TEST WELL #1"
    assert well_rt.company == "TEST CORP"
    assert well_rt.uwi == "1234567890"

    # Numerical verification
    np.testing.assert_allclose(well_rt["DEPT"], [1000.0, 1000.5, 1001.0])
    np.testing.assert_allclose(well_rt["GR"], [45.2, 52.1, 48.9])
    assert np.isnan(well_rt["NPHI"][1])
    assert pytest.approx(well_rt["NPHI"][0]) == 0.15
    assert pytest.approx(well_rt["NPHI"][2]) == 0.145


def test_write_to_file_and_roundtrip_las2() -> None:
    """Tests serializing a LAS 2.0 well to disk and reading it back."""
    well = turbolas.read("tests/data/4771-36-SESE.las")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "exported_well.las"
        res = turbolas.write(well, out_path, version=2.0)
        assert res is None
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Read back
        well_rt = turbolas.read(str(out_path))
        assert well_rt.num_rows == well.num_rows
        assert well_rt.num_curves == well.num_curves
        assert well_rt.curve_names == well.curve_names

        for name in well.curve_names:
            np.testing.assert_allclose(well[name], well_rt[name], equal_nan=True, atol=1e-3)


def test_write_las3_with_comma_delimiter() -> None:
    """Tests serializing a LAS 3.0 well with COMMA delimiter."""
    well = turbolas.read("tests/data/cwls_30_canonical.las")
    assert well.is_las3

    out_str = well.write(version=3.0, delimiter="COMMA")
    assert isinstance(out_str, str)
    assert "VERS.                        3.0" in out_str
    assert "DLM .                      COMMA" in out_str
    assert "~Log_Data | Log_Definition" in out_str

    well_rt = turbolas.read(out_str)
    assert well_rt.is_las3
    assert well_rt.delimiter == "COMMA"
    assert well_rt.num_rows == well.num_rows
    assert well_rt.num_curves == well.num_curves

    for name in well.curve_names:
        np.testing.assert_allclose(well[name], well_rt[name], equal_nan=True, atol=1e-3)


def test_write_las12_version() -> None:
    """Tests serializing with LAS 1.2 header format."""
    well = turbolas.read("tests/data/4771-36-SESE.las")
    out_str = well.write(version=1.2)
    assert isinstance(out_str, str)
    assert "VERS.   1.2" in out_str
    assert "~VERSION INFORMATION" in out_str

    well_rt = turbolas.read(out_str)
    assert well_rt.num_rows == well.num_rows
    assert well_rt.num_curves == well.num_curves


def test_write_wrapped_mode() -> None:
    """Tests serializing in wrapped data format."""
    well = turbolas.read("tests/data/4771-36-SESE.las")
    out_str = well.write(version=2.0, wrap=True)
    assert isinstance(out_str, str)
    assert "WRAP.   YES" in out_str

    well_rt = turbolas.read(out_str)
    assert well_rt.is_wrapped
    assert well_rt.num_rows == well.num_rows
    assert well_rt.num_curves == well.num_curves
    np.testing.assert_allclose(well["DEPT"], well_rt["DEPT"])
