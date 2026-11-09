"""Unit tests for permissive parsing of dirty, malformed legacy LAS files."""

from __future__ import annotations

import math

import numpy as np
import turbolas
from turbolas import ValidationMode


def test_missing_dot_in_mnemonic_permissive() -> None:
    """Tests recovering when header items omit the mandatory mnemonic dot."""
    dirty_las = (
        "~VERSION INFORMATION\n"
        " VERS. 2.0 : CWLS LOG ASCII STANDARD\n"
        " WRAP. NO : ONE LINE\n"
        "~WELL INFORMATION\n"
        " STRT 1000.0 : START DEPTH (NO DOT)\n"
        " STOP 1001.0 : STOP DEPTH (NO DOT)\n"
        " STEP 1.0 : STEP (NO DOT)\n"
        " NULL -999.25 : NULL (NO DOT)\n"
        " WELL ACME WILDCAT #1 : WELL NAME (NO DOT)\n"
        " COMP BIG OIL CORP : COMPANY (NO DOT)\n"
        "~CURVE INFORMATION\n"
        " DEPT 1 DEPTH (NO DOT OR COLON)\n"
        " GR 2 GAMMA RAY\n"
        "~A\n"
        " 1000.0  45.0\n"
        " 1001.0  50.0\n"
    )

    # In permissive mode, TurboLAS recovers gracefully!
    well = turbolas.read(dirty_las, validation_mode=ValidationMode.PERMISSIVE)
    assert well.num_rows == 2
    assert well.num_curves == 2
    assert well.well_name == "ACME WILDCAT #1"
    assert well.company == "BIG OIL CORP"
    assert well.start_depth == 1000.0
    assert well.stop_depth == 1001.0
    assert well.curve_names == ["DEPT", "GR"]
    np.testing.assert_allclose(well["DEPT"], [1000.0, 1001.0])
    np.testing.assert_allclose(well["GR"], [45.0, 50.0])


def test_asterisk_null_sentinels_and_error_tokens() -> None:
    """Tests handling ***, ERR, and non-numeric junk tokens in numeric columns."""
    dirty_las = (
        "~VERSION INFORMATION\n"
        " VERS. 2.0 : CWLS\n"
        " WRAP. NO : ONE LINE\n"
        "~WELL INFORMATION\n"
        " STRT.M 100.0 : START\n"
        " STOP.M 103.0 : STOP\n"
        " STEP.M 1.0 : STEP\n"
        " NULL. -999.25 : NULL\n"
        " WELL. DIRTY DATA TEST : WELL\n"
        "~CURVE INFORMATION\n"
        " DEPT.M : DEPTH\n"
        " GR.GAPI : GAMMA\n"
        " RHOB.K/M3 : DENSITY\n"
        "~A\n"
        " 100.0   45.0   ***\n"
        " 101.0   ERR    2.45\n"
        " 102.0   *****  BAD\n"
        " 103.0   50.0   -999.25\n"
    )

    well = turbolas.read(dirty_las, validation_mode=ValidationMode.PERMISSIVE)
    assert well.num_rows == 4
    assert well.num_curves == 3

    gr = well["GR"]
    assert gr[0] == 45.0
    assert math.isnan(gr[1])  # ERR -> NaN
    assert math.isnan(gr[2])  # ***** -> NaN
    assert gr[3] == 50.0

    rhob = well["RHOB"]
    assert math.isnan(rhob[0])  # *** -> NaN
    assert rhob[1] == 2.45
    assert math.isnan(rhob[2])  # BAD -> NaN
    assert math.isnan(rhob[3])  # -999.25 -> NaN


def test_spaces_after_tilde_and_unknown_sections() -> None:
    """Tests section headers with spaces like '~ WELL INFORMATION' and custom sections."""
    dirty_las = (
        "~ VERSION INFORMATION\n"
        " VERS. 2.0 : CWLS\n"
        " WRAP. NO : NO\n"
        "~   WELL INFORMATION\n"
        " STRT.M 100.0 : START\n"
        " STOP.M 100.0 : STOP\n"
        " STEP.M 0.0 : STEP\n"
        " NULL. -999.25 : NULL\n"
        " WELL. CUSTOM SECTION WELL : WELL\n"
        "~P_CALCULATED_PARAMETERS\n"
        " CONST. 3.1415 : PI\n"
        "~CURVE INFORMATION\n"
        " DEPT.M : DEPTH\n"
        " GR.GAPI : GAMMA\n"
        "~A\n"
        " 100.0  45.0\n"
    )

    well = turbolas.read(dirty_las, validation_mode=ValidationMode.PERMISSIVE)
    assert well.num_rows == 1
    assert well.num_curves == 2
    assert well.well_name == "CUSTOM SECTION WELL"
    np.testing.assert_allclose(well["DEPT"], [100.0])


def test_curve_names_on_ascii_delimiter_line() -> None:
    """Tests legacy files with curve mnemonics directly on the ~A line."""
    dirty_las = (
        "~VERSION INFORMATION\n"
        " VERS. 2.0 : CWLS\n"
        " WRAP. NO : NO\n"
        "~WELL INFORMATION\n"
        " STRT.M 100.0 : START\n"
        " STOP.M 101.0 : STOP\n"
        " STEP.M 1.0 : STEP\n"
        " NULL. -999.25 : NULL\n"
        " WELL. NO ~C SECTION : WELL\n"
        "~A  DEPT   GR   NPHI\n"
        " 100.0  45.0  0.15\n"
        " 101.0  50.0  0.18\n"
    )

    well = turbolas.read(dirty_las, validation_mode=ValidationMode.PERMISSIVE)
    assert well.num_rows == 2
    assert well.num_curves == 3
    assert well.curve_names == ["DEPT", "GR", "NPHI"]
    np.testing.assert_allclose(well["DEPT"], [100.0, 101.0])
    np.testing.assert_allclose(well["GR"], [45.0, 50.0])
    np.testing.assert_allclose(well["NPHI"], [0.15, 0.18])


def test_duplicate_curve_mnemonics_permissive() -> None:
    """Tests files with duplicate curve mnemonics without crashing."""
    dirty_las = (
        "~VERSION INFORMATION\n"
        " VERS. 2.0 : CWLS\n"
        " WRAP. NO : NO\n"
        "~WELL INFORMATION\n"
        " STRT.M 100.0 : START\n"
        " STOP.M 100.0 : STOP\n"
        " STEP.M 0.0 : STEP\n"
        " NULL. -999.25 : NULL\n"
        " WELL. DUPLICATE CURVES : WELL\n"
        "~CURVE INFORMATION\n"
        " DEPT.M : DEPTH\n"
        " GR.GAPI : MAIN PASS GAMMA\n"
        " GR.GAPI : REPEAT PASS GAMMA\n"
        "~A\n"
        " 100.0  45.0  47.5\n"
    )

    well = turbolas.read(dirty_las, validation_mode=ValidationMode.PERMISSIVE)
    assert well.num_rows == 1
    assert well.num_curves == 3
    # Both curves are accessible
    assert well.curve(1).description == "MAIN PASS GAMMA"
    assert well.curve(2).description == "REPEAT PASS GAMMA"
    assert well["GR"][0] == 45.0
    assert well["GR_2"][0] == 47.5
