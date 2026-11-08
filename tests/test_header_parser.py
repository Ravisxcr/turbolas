"""Comprehensive test suite for CWLS LAS header line parser."""

from __future__ import annotations

import pytest
from turbolas import (
    HeaderItem,
    HeaderParser,
    LASHeaderError,
    LASSectionError,
    SectionStateMachine,
    ValidationMode,
    parse_header_line,
)


class TestHeaderItemParsing:
    """Tests for parsing standard CWLS LAS header lines."""

    def test_version_records(self) -> None:
        """Parses ~V version information records."""
        line = "VERS. 2.0 : CWLS LOG ASCII STANDARD -VERSION 2.0"
        item = HeaderItem.parse_line(line)
        assert item.mnemonic == "VERS"
        assert item.unit == ""
        assert item.value == "2.0"
        assert item.description == "CWLS LOG ASCII STANDARD -VERSION 2.0"
        assert item.descr == item.description
        assert item.as_float() == 2.0
        assert not item.has_unit
        assert not item.is_empty_value
        assert item.has_description

        line_wrap = "WRAP. NO : ONE LINE PER DEPTH STEP"
        item_wrap = parse_header_line(line_wrap)
        assert item_wrap.mnemonic == "WRAP"
        assert item_wrap.unit == ""
        assert item_wrap.value == "NO"
        assert item_wrap.as_bool() is False

    def test_well_records(self) -> None:
        """Parses ~W well information records."""
        line_strt = "STRT .M 1670.0000 : START DEPTH"
        item_strt = HeaderItem.parse_line(line_strt)
        assert item_strt.mnemonic == "STRT"
        assert item_strt.unit == "M"
        assert item_strt.value == "1670.0000"
        assert item_strt.description == "START DEPTH"
        assert item_strt.as_float() == 1670.0
        assert item_strt.has_unit

        line_null = "NULL . -999.2500 : NULL VALUE"
        item_null = HeaderItem.parse_line(line_null)
        assert item_null.mnemonic == "NULL"
        assert item_null.unit == ""
        assert item_null.value == "-999.2500"
        assert item_null.as_float() == -999.25

        line_comp = "COMP . ANY OIL COMPANY INC. : COMPANY"
        item_comp = HeaderItem.parse_line(line_comp)
        assert item_comp.mnemonic == "COMP"
        assert item_comp.unit == ""
        assert item_comp.value == "ANY OIL COMPANY INC."
        assert item_comp.description == "COMPANY"

        line_loc = "LOC . 10-20-45-20W4 : LOCATION"
        item_loc = HeaderItem.parse_line(line_loc)
        assert item_loc.mnemonic == "LOC"
        assert item_loc.value == "10-20-45-20W4"

    def test_curve_records(self) -> None:
        """Parses ~C curve definition records."""
        line_dept = "DEPT .M : 1 DEPTH"
        item_dept = HeaderItem.parse_line(line_dept)
        assert item_dept.mnemonic == "DEPT"
        assert item_dept.unit == "M"
        assert item_dept.value == ""
        assert item_dept.is_empty_value
        assert item_dept.description == "1 DEPTH"

        line_rhob = "RHOB .K/M3 2550.0000 : 2 BULK DENSITY"
        item_rhob = HeaderItem.parse_line(line_rhob)
        assert item_rhob.mnemonic == "RHOB"
        assert item_rhob.unit == "K/M3"
        assert item_rhob.value == "2550.0000"
        assert item_rhob.as_float() == 2550.0
        assert item_rhob.description == "2 BULK DENSITY"

    def test_complex_units(self) -> None:
        """Parses various technical measurement units."""
        cases = [
            ("RES.OHM-M 10.5 : RES", "RES", "OHM-M", "10.5"),
            ("TEMP.DEG_C 85.0 : TEMP", "TEMP", "DEG_C", "85.0"),
            ("POR.V/V 0.25 : POROSITY", "POR", "V/V", "0.25"),
            ("CALI.IN 8.5 : CALIPER", "CALI", "IN", "8.5"),
            ("PERM.MD 150 : PERM", "PERM", "MD", "150"),
            ("GAS.% 12.5 : GAS CONC", "GAS", "%", "12.5"),
            ("DT.US/M 60.5 : SONIC", "DT", "US/M", "60.5"),
        ]
        for line, exp_m, exp_u, exp_v in cases:
            item = HeaderItem.parse_line(line)
            assert item.mnemonic == exp_m
            assert item.unit == exp_u
            assert item.value == exp_v

    def test_empty_fields(self) -> None:
        """Handles empty values, empty units, and empty descriptions."""
        item = HeaderItem.parse_line("EMPTY. :")
        assert item.mnemonic == "EMPTY"
        assert item.unit == ""
        assert item.value == ""
        assert item.description == ""
        assert item.is_empty_value
        assert not item.has_unit
        assert not item.has_description

        item_unit_only = HeaderItem.parse_line("DEPT.M :")
        assert item_unit_only.mnemonic == "DEPT"
        assert item_unit_only.unit == "M"
        assert item_unit_only.value == ""
        assert item_unit_only.description == ""

        # No space before colon
        item_compact = HeaderItem.parse_line("DEPT.M:DEPTH")
        assert item_compact.mnemonic == "DEPT"
        assert item_compact.unit == "M"
        assert item_compact.value == ""
        assert item_compact.description == "DEPTH"


class TestColonHandling:
    """Tests for complex colon parsing (timestamps, colons in descriptions, ratios)."""

    def test_timestamp_in_value(self) -> None:
        """Ensures colons in HH:MM:SS values are not treated as description separators."""
        line = "TIME . 12:30:00 : LOGGING TIME"
        item = HeaderItem.parse_line(line)
        assert item.mnemonic == "TIME"
        assert item.unit == ""
        assert item.value == "12:30:00"
        assert item.description == "LOGGING TIME"

        line_short = "TIME . 12:30 : LOGGING TIME"
        item_short = HeaderItem.parse_line(line_short)
        assert item_short.value == "12:30"
        assert item_short.description == "LOGGING TIME"

        line_iso = "DATE . 2024-01-01 14:30:00 : LOG DATE AND TIME"
        item_iso = HeaderItem.parse_line(line_iso)
        assert item_iso.value == "2024-01-01 14:30:00"
        assert item_iso.description == "LOG DATE AND TIME"

    def test_colons_in_description(self) -> None:
        """Ensures colons inside the description field are preserved intact."""
        line = "WELL . A_1 : NOTE: DRILL HOLE #1"
        item = HeaderItem.parse_line(line)
        assert item.mnemonic == "WELL"
        assert item.value == "A_1"
        assert item.description == "NOTE: DRILL HOLE #1"

        line_multi = "COMM . : PART 1: TEST; PART 2: FINAL"
        item_multi = HeaderItem.parse_line(line_multi)
        assert item_multi.mnemonic == "NOTE" or item_multi.mnemonic == "COMM"
        assert item_multi.value == ""
        assert item_multi.description == "PART 1: TEST; PART 2: FINAL"

    def test_timestamp_in_value_and_colon_in_description(self) -> None:
        """Correctly separates value and description when both contain colons."""
        line = "TIME . 12:30:00 : RUN TIME: 12:30:00"
        item = HeaderItem.parse_line(line)
        assert item.mnemonic == "TIME"
        assert item.value == "12:30:00"
        assert item.description == "RUN TIME: 12:30:00"

    def test_ratio_in_value(self) -> None:
        """Preserves numeric ratios with internal colons in value field."""
        line = "SCALE . 1:24000 : MAP SCALE"
        item = HeaderItem.parse_line(line)
        assert item.mnemonic == "SCALE"
        assert item.value == "1:24000"
        assert item.description == "MAP SCALE"

    def test_quoted_values(self) -> None:
        """Handles quoted values containing colons and spaces."""
        line = 'WELL . "BAKER: #1" : WELL NAME'
        item = HeaderItem.parse_line(line)
        assert item.mnemonic == "WELL"
        assert item.value == '"BAKER: #1"'
        assert item.unquoted_value() == "BAKER: #1"
        assert item.description == "WELL NAME"


class TestValidationModes:
    """Tests for strict and permissive mode behaviors."""

    def test_strict_rejects_missing_dot(self) -> None:
        """Strict mode rejects lines missing the dot separator."""
        with pytest.raises(LASHeaderError, match="missing required dot"):
            HeaderItem.parse_line("STRT M 1670.0000 : START DEPTH")

        with pytest.raises(LASHeaderError, match="missing required dot"):
            HeaderItem.parse_line("NO_DOT_HERE : DESC")

    def test_strict_rejects_empty_mnemonic(self) -> None:
        """Strict mode rejects lines with empty mnemonic."""
        with pytest.raises(LASHeaderError, match="empty mnemonic"):
            HeaderItem.parse_line(".M 1670.0000 : START DEPTH")

        with pytest.raises(LASHeaderError, match="empty mnemonic"):
            HeaderItem.parse_line("   .M 1670.0000 : START DEPTH")

    def test_strict_rejects_missing_colon(self) -> None:
        """Strict mode rejects lines without a description delimiter colon."""
        with pytest.raises(LASHeaderError, match="missing required colon"):
            HeaderItem.parse_line("STRT .M 1670.0000")

    def test_permissive_allows_missing_colon(self) -> None:
        """Permissive mode accepts lines without colon and sets empty description."""
        item = HeaderItem.parse_line("STRT .M 1670.0000", mode=ValidationMode.Permissive)
        assert item.mnemonic == "STRT"
        assert item.unit == "M"
        assert item.value == "1670.0000"
        assert item.description == ""

    def test_whitespace_in_mnemonic(self) -> None:
        """Strict mode rejects spaces in mnemonic; permissive mode allows them."""
        line = "SET NAME . 123 : SET IDENTIFIER"
        with pytest.raises(LASHeaderError, match="mnemonic contains whitespace"):
            HeaderItem.parse_line(line, mode=ValidationMode.Strict)

        item = HeaderItem.parse_line(line, mode=ValidationMode.Permissive)
        assert item.mnemonic == "SET NAME"
        assert item.value == "123"

    def test_rejects_comments_and_delimiters(self) -> None:
        """Comments, section delimiters, and blanks are not header lines."""
        with pytest.raises(LASHeaderError, match="found comment"):
            HeaderItem.parse_line("# This is a comment")

        with pytest.raises(LASHeaderError, match="found section delimiter"):
            HeaderItem.parse_line("~WELL INFORMATION")

        with pytest.raises(LASHeaderError, match="empty or whitespace"):
            HeaderItem.parse_line("   ")

    def test_try_parse_line(self) -> None:
        """`try_parse_line` returns None on errors instead of raising."""
        assert HeaderItem.try_parse_line("INVALID LINE") is None
        assert HeaderItem.try_parse_line("# COMMENT") is None
        valid = HeaderItem.try_parse_line("STRT.M 100 : DEPTH")
        assert valid is not None
        assert valid.mnemonic == "STRT"


class TestTypeConversionsAndFormatting:
    """Tests for typed conversions and string formatting."""

    def test_conversions(self) -> None:
        """Tests as_float, as_int, and as_bool methods."""
        item_num = HeaderItem("VAL", "", "123.456", "NUM")
        assert item_num.as_float() == 123.456
        assert item_num.as_int() is None

        item_int = HeaderItem("COUNT", "", "42", "COUNT")
        assert item_int.as_int() == 42
        assert item_int.as_float() == 42.0

        item_bool_t = HeaderItem("FLAG", "", "YES", "FLAG")
        assert item_bool_t.as_bool() is True
        item_bool_f = HeaderItem("FLAG", "", "NO", "FLAG")
        assert item_bool_f.as_bool() is False

        item_one = HeaderItem("FLAG", "", "1", "FLAG")
        assert item_one.as_bool() is True
        item_zero = HeaderItem("FLAG", "", "0", "FLAG")
        assert item_zero.as_bool() is False

    def test_formatting_roundtrip(self) -> None:
        """Formats item and verifies roundtrip parsing."""
        item = HeaderItem("STRT", "M", "1670.0000", "START DEPTH")
        line = item.to_line()
        assert "STRT" in line
        assert ".M" in line
        assert "1670.0000" in line
        assert ": START DEPTH" in line

        roundtrip = HeaderItem.parse_line(line)
        assert roundtrip.mnemonic == item.mnemonic
        assert roundtrip.unit == item.unit
        assert roundtrip.value == item.value
        assert roundtrip.description == item.description
        assert str(item) == line
        assert repr(item).startswith("HeaderItem(mnemonic='STRT'")

    def test_equality(self) -> None:
        """HeaderItems with identical fields compare equal."""
        item1 = HeaderItem("STRT", "M", "100", "DEPTH")
        item2 = HeaderItem("STRT", "M", "100", "DEPTH")
        item3 = HeaderItem("STOP", "M", "100", "DEPTH")
        assert item1 == item2
        assert item1 != item3


class TestHeaderParser:
    """Tests for batch HeaderParser class."""

    def test_batch_parse_lines(self) -> None:
        """Parses a list of lines, automatically skipping comments and blanks."""
        parser = HeaderParser(strict=True)
        assert parser.is_strict
        assert not parser.is_permissive

        lines = [
            "# Well Information Section",
            "",
            "STRT .M 1670.0 : START DEPTH",
            "# Next parameter",
            "STOP .M 1660.0 : STOP DEPTH",
            "STEP .M -0.125 : STEP",
            "",
        ]

        items = parser.parse_lines(lines, start_line_number=10)
        assert len(items) == 3
        assert items[0].mnemonic == "STRT"
        assert items[0].line_number == 12
        assert items[1].mnemonic == "STOP"
        assert items[1].line_number == 14
        assert items[2].mnemonic == "STEP"
        assert items[2].line_number == 15

    def test_parser_mode_switch(self) -> None:
        """Validates changing parser mode between strict and permissive."""
        parser = HeaderParser()
        assert parser.is_strict

        # Strict rejects missing colon
        with pytest.raises(LASHeaderError):
            parser.parse_line("STRT .M 100")

        # Switch to permissive
        parser.mode = ValidationMode.PERMISSIVE
        assert parser.is_permissive
        item = parser.parse_line("STRT .M 100")
        assert item.mnemonic == "STRT"
        assert item.value == "100"


class TestStateMachineIntegration:
    """Tests parse_header_line via SectionStateMachine."""

    def test_state_machine_header_line_parsing(self) -> None:
        """StateMachine parses header lines when inside header sections."""
        sm = SectionStateMachine(strict=True)

        # Before any section: error
        with pytest.raises(LASSectionError, match="not currently in any section"):
            sm.parse_header_line("VERS. 2.0 : CWLS LAS")

        # Enter ~V
        sm.feed_line("~VERSION")
        item_v = sm.parse_header_line("VERS. 2.0 : CWLS LOG ASCII STANDARD")
        assert item_v.mnemonic == "VERS"
        assert item_v.value == "2.0"

        # Enter ~W
        sm.feed_line("~WELL")
        item_w = sm.parse_header_line("STRT.M 100.0 : START DEPTH")
        assert item_w.mnemonic == "STRT"
        assert item_w.unit == "M"

        # Enter ~C
        sm.feed_line("~CURVE")
        item_c = sm.parse_header_line("DEPT.M : 1 DEPTH")
        assert item_c.mnemonic == "DEPT"

        # Enter ~A (data section - not a header section!)
        sm.feed_line("~ASCII")
        with pytest.raises(LASSectionError, match="only header sections"):
            sm.parse_header_line("100.0 200.0 300.0")
