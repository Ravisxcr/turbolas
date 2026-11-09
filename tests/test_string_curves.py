"""Unit tests for string / categorical curve support in CWLS LAS 3.0."""

from __future__ import annotations

import polars as pl
import pytest
import turbolas


def test_las3_string_lithology_columns() -> None:
    """Tests parsing string/text curves such as lithology and descriptions."""
    las_text = (
        "~Version Information\n"
        " VERS. 3.0 : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
        " WRAP. NO : ONE LINE PER DEPTH STEP\n"
        " DLM . COMMA : DELIMITING CHARACTER\n"
        "~Well Information\n"
        " STRT.M 1200.0 : START DEPTH\n"
        " STOP.M 1202.0 : STOP DEPTH\n"
        " STEP.M 1.0 : STEP\n"
        " NULL. -999.25 : NULL\n"
        " WELL. STRING TEST WELL : WELL NAME\n"
        "~Log_Definition\n"
        " DEPT.M : Depth\n"
        " LITH. : Lithology description\n"
        " PORO.V/V : Porosity\n"
        " QUAL. : Quality tag\n"
        "~Log_Data | Log_Definition\n"
        '1200.0,"FINE SANDSTONE",0.18,A\n'
        "1201.0,SHALE,0.05,B\n"
        '1202.0,"FOSSILIFEROUS LIMESTONE",0.12,A\n'
    )

    well = turbolas.read(las_text)
    assert well.num_rows == 3
    assert well.num_curves == 4
    assert well.curve_names == ["DEPT", "LITH", "PORO", "QUAL"]

    # DEPT and PORO are numeric
    dept_curve = well.curve("DEPT")
    assert not dept_curve.is_string
    assert pytest.approx(well["DEPT"][0]) == 1200.0

    poro_curve = well.curve("PORO")
    assert not poro_curve.is_string
    assert pytest.approx(well["PORO"][0]) == 0.18

    # LITH is string/categorical
    lith_curve = well.curve("LITH")
    assert lith_curve.is_string
    assert lith_curve.num_rows == 3
    assert lith_curve.string_data() == ["FINE SANDSTONE", "SHALE", "FOSSILIFEROUS LIMESTONE"]
    assert lith_curve.data == ["FINE SANDSTONE", "SHALE", "FOSSILIFEROUS LIMESTONE"]
    assert well["LITH"] == ["FINE SANDSTONE", "SHALE", "FOSSILIFEROUS LIMESTONE"]
    assert well["LITH"][0] == "FINE SANDSTONE"
    assert well["LITH"][1] == "SHALE"
    assert repr(lith_curve) == "<LasCurve LITH [TEXT]: 3 samples>"

    # QUAL is string
    qual_curve = well.curve("QUAL")
    assert qual_curve.is_string
    assert well["QUAL"] == ["A", "B", "A"]

    # Native dictionary export preserves string types
    d = well.to_dict()
    assert d["LITH"] == ["FINE SANDSTONE", "SHALE", "FOSSILIFEROUS LIMESTONE"]
    assert d["QUAL"] == ["A", "B", "A"]

    # Ingestion into Polars DataFrame produces String columns!
    df = pl.DataFrame(d)
    assert df.schema["LITH"] == pl.String
    assert df.schema["QUAL"] == pl.String
    assert df.schema["DEPT"] == pl.Float64
    assert df.schema["PORO"] == pl.Float64
    assert df["LITH"].to_list() == ["FINE SANDSTONE", "SHALE", "FOSSILIFEROUS LIMESTONE"]


def test_las3_string_curve_roundtrip_write() -> None:
    """Tests writing and reading back a file containing string columns."""
    las_text = (
        "~Version Information\n"
        " VERS. 3.0 : CWLS LOG ASCII STANDARD -VERSION 3.0\n"
        " WRAP. NO : ONE LINE PER DEPTH STEP\n"
        " DLM . COMMA : DELIMITING CHARACTER\n"
        "~Well Information\n"
        " STRT.M 100.0 : START DEPTH\n"
        " STOP.M 101.0 : STOP DEPTH\n"
        " STEP.M 1.0 : STEP\n"
        " NULL. -999.25 : NULL\n"
        " WELL. ROUNDTRIP TEST : WELL NAME\n"
        "~Log_Definition\n"
        " DEPT.M : Depth\n"
        " FACIES. : Facies Name\n"
        "~Log_Data | Log_Definition\n"
        '100.0,"CHANNEL SAND"\n'
        '101.0,"FLOODPLAIN SILT"\n'
    )

    well = turbolas.read(las_text)
    serialized = well.write(version=3.0, delimiter="COMMA")
    assert isinstance(serialized, str)
    assert "CHANNEL SAND" in serialized
    assert "FLOODPLAIN SILT" in serialized

    well_rt = turbolas.read(serialized)
    assert well_rt.num_rows == 2
    assert well_rt.curve("FACIES").is_string
    assert well_rt["FACIES"] == ["CHANNEL SAND", "FLOODPLAIN SILT"]
