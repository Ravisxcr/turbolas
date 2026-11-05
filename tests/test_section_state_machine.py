"""Comprehensive test suite for CWLS section delimiter state machine."""

import pytest
from turbolas import (
    LASSectionError,
    SectionDelimiter,
    SectionStateMachine,
    SectionType,
    State,
    ValidationMode,
)


def test_section_types_enum() -> None:
    """Ensure all six CWLS LAS section types are defined with correct properties."""
    sections = [
        (SectionType.VERSION, "V", "~V", "VERSION", True, False, True, False, False),
        (SectionType.WELL, "W", "~W", "WELL", True, False, True, False, False),
        (SectionType.CURVE, "C", "~C", "CURVE", True, False, True, False, False),
        (SectionType.PARAMETER, "P", "~P", "PARAMETER", False, True, True, False, False),
        (SectionType.OTHER, "O", "~O", "OTHER", False, True, False, False, True),
        (SectionType.ASCII, "A", "~A", "ASCII", True, False, False, True, False),
    ]

    for (
        sec,
        char_code,
        delim,
        name,
        mandatory,
        optional,
        header,
        data,
        remarks,
    ) in sections:
        assert sec.char_code == char_code
        assert sec.delimiter == delim
        assert sec.name == name
        assert sec.is_mandatory == mandatory
        assert sec.is_optional == optional
        assert sec.is_header_section == header
        assert sec.is_data_section == data
        assert sec.is_remarks_section == remarks


def test_section_type_casing_aliases() -> None:
    """Ensure both uppercase and PascalCase access works identically."""
    assert SectionType.VERSION == SectionType.Version
    assert SectionType.WELL == SectionType.Well
    assert SectionType.CURVE == SectionType.Curve
    assert SectionType.PARAMETER == SectionType.Parameter
    assert SectionType.OTHER == SectionType.Other
    assert SectionType.ASCII == SectionType.Ascii


def test_state_enum() -> None:
    """Ensure State enum members and properties operate correctly."""
    assert State.INITIAL == State.Initial
    assert State.VERSION == State.Version
    assert State.COMPLETE == State.Complete

    assert State.INITIAL.is_initial is True
    assert State.INITIAL.is_in_section is False
    assert State.INITIAL.section_type is None

    assert State.VERSION.is_in_section is True
    assert State.VERSION.section_type == SectionType.VERSION

    assert State.ASCII.is_in_data is True

    assert State.COMPLETE.is_complete is True
    assert State.COMPLETE.is_in_section is False


def test_section_delimiter_classification() -> None:
    """Test line classification helper methods."""
    assert SectionDelimiter.is_delimiter("~VERSION") is True
    assert SectionDelimiter.is_delimiter("   ~V   ") is True
    assert SectionDelimiter.is_delimiter("# comment") is False
    assert SectionDelimiter.is_delimiter("STRT.M 100.0:") is False

    assert SectionDelimiter.is_comment("# comment") is True
    assert SectionDelimiter.is_comment("   # indented comment") is True
    assert SectionDelimiter.is_comment("~V") is False

    assert SectionDelimiter.is_blank("") is True
    assert SectionDelimiter.is_blank("   \t  \r\n") is True
    assert SectionDelimiter.is_blank("~V") is False


def test_section_delimiter_parsing() -> None:
    """Test parsing section headers into SectionDelimiter objects."""
    d_v = SectionDelimiter.parse_line("~VERSION INFORMATION", line_number=1)
    assert d_v.section_type == SectionType.VERSION
    assert d_v.char_code == "V"
    assert d_v.line_number == 1
    assert d_v.title == "INFORMATION"

    d_w = SectionDelimiter.parse_line("~Well Information Section", line_number=5)
    assert d_w.section_type == SectionType.WELL
    assert d_w.char_code == "W"
    assert d_w.title == "Information Section"

    d_c = SectionDelimiter.parse_line("~CURVE INFORMATION", line_number=15)
    assert d_c.section_type == SectionType.CURVE
    assert d_c.char_code == "C"

    d_p = SectionDelimiter.parse_line("~PARAMETER INFORMATION", line_number=25)
    assert d_p.section_type == SectionType.PARAMETER
    assert d_p.char_code == "P"

    d_o = SectionDelimiter.parse_line("~OTHER", line_number=35)
    assert d_o.section_type == SectionType.OTHER
    assert d_o.char_code == "O"
    assert d_o.title == ""

    d_a = SectionDelimiter.parse_line("~A DEPTH RHOB NPHI", line_number=45)
    assert d_a.section_type == SectionType.ASCII
    assert d_a.char_code == "A"
    assert d_a.title == "DEPTH RHOB NPHI"


def test_section_delimiter_errors() -> None:
    """Test that invalid delimiters raise LASSectionError (inheriting from ValueError)."""
    with pytest.raises(LASSectionError):
        SectionDelimiter.parse_line("not a delimiter")

    with pytest.raises(LASSectionError):
        SectionDelimiter.parse_line("~")

    with pytest.raises(LASSectionError):
        SectionDelimiter.parse_line("~Z UNKNOWN")


def test_strict_cwls_state_machine_full_run() -> None:
    """Simulate a complete standard CWLS LAS 2.0 file in strict mode."""
    sm = SectionStateMachine(strict=True)

    assert sm.current_state == State.INITIAL
    assert sm.current_section is None
    assert sm.line_number == 0

    # Comments and blanks prior to ~V are allowed
    a1 = sm.process_line("# CWLS Log ASCII Standard - File")
    assert a1.is_comment is True
    assert a1.text == "CWLS Log ASCII Standard - File"

    a2 = sm.process_line("")
    assert a2.is_blank is True

    # 1. ~V
    a3 = sm.process_line("~VERSION INFORMATION")
    assert a3.is_section_start is True
    assert a3.section == SectionType.VERSION
    assert sm.current_state == State.VERSION
    assert sm.current_section == SectionType.VERSION

    sm.process_line(" VERS. 2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0")
    sm.process_line(" WRAP. NO  : ONE LINE PER RECORD STEP")
    assert sm.current_section_line_count == 2
    assert sm.get_line_count_for_section(SectionType.VERSION) == 2

    # 2. ~W
    sm.process_line("~WELL INFORMATION")
    assert sm.current_state == State.WELL
    sm.process_line(" STRT.M 1670.0 : START DEPTH")
    sm.process_line(" STOP.M 1660.0 : STOP DEPTH")
    sm.process_line(" STEP.M -0.125 : STEP")
    sm.process_line(" NULL.  -999.25: NULL VALUE")
    assert sm.get_line_count_for_section(SectionType.WELL) == 4

    # 3. ~C
    sm.process_line("~CURVE INFORMATION")
    assert sm.current_state == State.CURVE
    sm.process_line(" DEPT.M   : 1 DEPTH")
    sm.process_line(" RHOB.K/M3: 2 BULK DENSITY")
    assert sm.get_line_count_for_section(SectionType.CURVE) == 2

    # 4. ~P (optional)
    sm.process_line("~PARAMETER INFORMATION")
    assert sm.current_state == State.PARAMETER
    sm.process_line(" BHT.DEGC 35.5 : BOTTOM HOLE TEMP")
    assert sm.get_line_count_for_section(SectionType.PARAMETER) == 1

    # 5. ~O (optional)
    sm.process_line("~OTHER")
    assert sm.current_state == State.OTHER
    sm.process_line("Tool calibration on 15-DEC-1988")
    assert sm.get_line_count_for_section(SectionType.OTHER) == 1

    # 6. ~A
    sm.process_line("~A DEPTH RHOB")
    assert sm.current_state == State.ASCII
    assert sm.is_in_data_section is True
    sm.process_line("1670.000 2550.00")
    sm.process_line("1669.875 2548.00")
    assert sm.get_line_count_for_section(SectionType.ASCII) == 2

    # Verify tracking
    assert sm.section_count == 6
    assert sm.has_seen_section(SectionType.VERSION) is True
    assert sm.has_seen_section(SectionType.ASCII) is True
    assert sm.seen_sections == [
        SectionType.VERSION,
        SectionType.WELL,
        SectionType.CURVE,
        SectionType.PARAMETER,
        SectionType.OTHER,
        SectionType.ASCII,
    ]

    # Finish
    sm.finish()
    assert sm.is_complete is True
    assert sm.current_state == State.COMPLETE


def test_strict_rejects_non_version_first() -> None:
    """Strict mode must reject any first section other than ~V."""
    sm = SectionStateMachine(strict=True)
    with pytest.raises(LASSectionError, match="must begin with ~V"):
        sm.process_line("~WELL INFORMATION")

    sm2 = SectionStateMachine(strict=True)
    with pytest.raises(LASSectionError, match="must begin with ~V"):
        sm2.process_line("~A")


def test_strict_rejects_content_before_first_section() -> None:
    """Strict mode must reject non-blank, non-comment lines before first section."""
    sm = SectionStateMachine(strict=True)
    with pytest.raises(LASSectionError, match="unexpected content before first section"):
        sm.process_line("STRT.M 100.0:")


def test_strict_rejects_duplicate_sections() -> None:
    """Strict mode must reject duplicate sections."""
    sm = SectionStateMachine(strict=True)
    sm.process_line("~VERSION")
    sm.process_line("~WELL")
    with pytest.raises(LASSectionError, match="duplicate section"):
        sm.process_line("~VERSION")


def test_strict_rejects_ascii_before_curve() -> None:
    """Strict mode must reject ~A before ~C has been defined."""
    sm = SectionStateMachine(strict=True)
    sm.process_line("~VERSION")
    sm.process_line("~WELL")
    with pytest.raises(LASSectionError, match="cannot appear before Curve section"):
        sm.process_line("~A")


def test_strict_rejects_section_after_ascii() -> None:
    """Strict mode must reject any section appearing after ~A."""
    sm = SectionStateMachine(strict=True)
    sm.process_line("~VERSION")
    sm.process_line("~WELL")
    sm.process_line("~CURVE")
    sm.process_line("~A")
    with pytest.raises(LASSectionError, match="cannot appear after ASCII log data section"):
        sm.process_line("~OTHER")


def test_strict_finish_missing_required() -> None:
    """finish() in strict mode must verify mandatory sections."""
    sm = SectionStateMachine(strict=True)
    sm.process_line("~VERSION")
    sm.process_line("~WELL")
    # Curve and Ascii missing
    with pytest.raises(LASSectionError, match="Missing required CWLS LAS section"):
        sm.finish()


def test_permissive_mode() -> None:
    """Permissive mode allows flexible / non-standard ordering and duplication."""
    sm = SectionStateMachine(mode=ValidationMode.Permissive)

    # Content before first section is skipped
    sm.process_line("stray non-comment line")
    assert sm.current_state == State.INITIAL

    # Can start with ~W
    sm.process_line("~WELL")
    assert sm.current_state == State.WELL

    # Can transition to ~A without ~C
    sm.process_line("~A")
    assert sm.current_state == State.ASCII

    # Can transition after ~A
    sm.process_line("~VERSION")
    assert sm.current_state == State.VERSION

    # Can have duplicate sections
    sm.process_line("~WELL")
    assert sm.current_state == State.WELL

    # finish() succeeds even if Curve was never seen
    sm.finish()
    assert sm.is_complete is True


def test_feed_line_helper() -> None:
    """Ensure feed_line returns SectionType only when a section begins."""
    sm = SectionStateMachine(strict=True)
    assert sm.feed_line("# comment") is None
    assert sm.feed_line("~VERSION") == SectionType.VERSION
    assert sm.feed_line("VERS. 2.0:") is None
    assert sm.feed_line("~WELL") == SectionType.WELL


def test_reset() -> None:
    """Ensure reset() restores the state machine to Initial cleanly."""
    sm = SectionStateMachine(strict=True)
    sm.process_line("~VERSION")
    sm.process_line("VERS. 2.0:")
    assert sm.section_count == 1

    sm.reset()
    assert sm.current_state == State.INITIAL
    assert sm.line_number == 0
    assert sm.section_count == 0
    assert sm.get_line_count_for_section(SectionType.VERSION) == 0
