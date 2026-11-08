"""Comprehensive test suite for dynamic NULL sentinel detection and IEEE 754 NaN mapping."""

from __future__ import annotations

import math

import pytest
from turbolas import (
    HeaderItem,
    LASSectionError,
    NullPolicy,
    NullSentinel,
    SectionStateMachine,
    is_null,
    map_from_nan,
    map_to_nan,
)


class TestNullSentinelBasics:
    """Tests for NullSentinel construction, configuration, and defaults."""

    def test_default_cwls(self) -> None:
        """Default CWLS sentinel has value -999.25 and strict policy."""
        sentinel = NullSentinel.default_cwls()
        assert sentinel.value == -999.25
        assert sentinel.policy == NullPolicy.Strict
        assert sentinel.policy.is_strict
        assert not sentinel.policy.is_common
        assert sentinel.tolerance == 1e-5
        assert str(sentinel) == "-999.25"

    def test_custom_sentinel(self) -> None:
        """Constructs NullSentinel with custom value, policy, and tolerance."""
        sentinel = NullSentinel(value=-9999.0, policy=NullPolicy.Common, tolerance=1e-4)
        assert sentinel.value == -9999.0
        assert sentinel.policy == NullPolicy.Common
        assert sentinel.policy.is_common
        assert sentinel.tolerance == 1e-4

        # Mutation via setters
        sentinel.value = -32767.0
        assert sentinel.value == -32767.0
        sentinel.policy = NullPolicy.Strict
        assert sentinel.policy.is_strict
        sentinel.tolerance = 1e-6
        assert sentinel.tolerance == 1e-6

    def test_null_policy_enum_variants_and_aliases(self) -> None:
        """Tests NullPolicy enum variants and uppercase aliases."""
        assert NullPolicy.STRICT == NullPolicy.Strict
        assert NullPolicy.COMMON == NullPolicy.Common
        assert NullPolicy.AGGRESSIVE == NullPolicy.Aggressive
        assert NullPolicy.NONE == NullPolicy.None_


class TestIEEE754Mapping:
    """Tests for mapping between NULL sentinels and IEEE 754 NaN."""

    def test_is_null_and_map_to_nan(self) -> None:
        """Checks is_null predicate and NaN mapping for floats."""
        sentinel = NullSentinel.default_cwls()

        # Sentinel maps to NaN
        assert sentinel.is_null(-999.25)
        mapped = sentinel.map_to_nan(-999.25)
        assert math.isnan(mapped)

        # Existing NaN is recognized as null and remains NaN
        assert sentinel.is_null(float("nan"))
        assert math.isnan(sentinel.map_to_nan(float("nan")))

        # Valid non-null data remains unchanged
        assert not sentinel.is_null(1670.0)
        assert sentinel.map_to_nan(1670.0) == 1670.0

        assert not sentinel.is_null(0.0)
        assert sentinel.map_to_nan(0.0) == 0.0

    def test_map_from_nan(self) -> None:
        """Replaces IEEE 754 NaN with the sentinel value."""
        sentinel = NullSentinel(-999.25)
        assert sentinel.map_from_nan(float("nan")) == -999.25
        assert sentinel.map_from_nan(1670.0) == 1670.0

    def test_tolerance_comparison(self) -> None:
        """Tests float comparison with tolerance epsilon."""
        sentinel = NullSentinel(-999.25, tolerance=1e-5)
        # Very close value matches sentinel
        assert sentinel.is_null(-999.250001)
        assert math.isnan(sentinel.map_to_nan(-999.250001))

        # Different value does not match
        assert not sentinel.is_null(-999.20)
        assert sentinel.map_to_nan(-999.20) == -999.20

    def test_string_token_mapping(self) -> None:
        """Tests parsing and checking string tokens."""
        sentinel = NullSentinel.default_cwls()

        # String representations of sentinel
        assert sentinel.is_null_str("-999.25")
        assert sentinel.is_null_str("-999.2500")
        assert sentinel.is_null_str("NaN")
        assert sentinel.is_null_str("nan")
        assert sentinel.is_null_str("NULL")
        assert sentinel.is_null_str("")
        assert not sentinel.is_null_str("1670.0")

        # map_str_to_nan
        assert math.isnan(sentinel.map_str_to_nan("-999.25"))  # type: ignore[arg-type]
        assert math.isnan(sentinel.map_str_to_nan("NaN"))  # type: ignore[arg-type]
        assert sentinel.map_str_to_nan("1670.5") == 1670.5
        assert sentinel.map_str_to_nan("INVALID_TOKEN") is None


class TestPolicies:
    """Tests for Strict, Common, Aggressive, and None policies."""

    def test_strict_policy(self) -> None:
        """Strict policy only matches the declared sentinel value and NaN."""
        sentinel = NullSentinel(-9999.0, policy=NullPolicy.Strict)
        assert sentinel.is_null(-9999.0)
        assert sentinel.is_null(float("nan"))
        # Does NOT match -999.25 because file declared -9999.0
        assert not sentinel.is_null(-999.25)

    def test_common_policy(self) -> None:
        """Common policy matches declared sentinel plus common LAS sentinels."""
        sentinel = NullSentinel(-9999.0, policy=NullPolicy.Common)
        assert sentinel.is_null(-9999.0)
        assert sentinel.is_null(-999.25)
        assert sentinel.is_null(999.25)
        assert sentinel.is_null(-999.0)

    def test_aggressive_policy(self) -> None:
        """Aggressive policy matches common and legacy/vendor sentinels."""
        sentinel = NullSentinel(-999.25, policy=NullPolicy.Aggressive)
        assert sentinel.is_null(-999.25)
        assert sentinel.is_null(-32767.0)
        assert sentinel.is_null(32767.0)
        assert sentinel.is_null(-999.99)
        assert sentinel.is_null(-0.0)

    def test_none_policy(self) -> None:
        """None policy disables sentinel mapping (only literal NaN is null)."""
        sentinel = NullSentinel(-999.25, policy=NullPolicy.None_)
        assert not sentinel.is_null(-999.25)
        assert sentinel.map_to_nan(-999.25) == -999.25
        assert sentinel.is_null(float("nan"))

    def test_custom_substitute(self) -> None:
        """Adds custom substitute values."""
        sentinel = NullSentinel(-999.25, policy=NullPolicy.Strict)
        assert not sentinel.is_null(-888.8)
        sentinel.add_substitute(-888.8)
        assert sentinel.is_null(-888.8)
        assert -888.8 in sentinel.custom_substitutes


class TestArrayAndLineOperations:
    """Tests for list/slice replacements and ASCII data line parsing."""

    def test_replace_nulls_and_nans(self) -> None:
        """Tests batch replacement of nulls to NaNs and NaNs to sentinels."""
        sentinel = NullSentinel.default_cwls()
        data = [100.0, -999.25, 200.0, -999.2500, 300.0]
        mapped = sentinel.replace_nulls(data)
        assert mapped[0] == 100.0
        assert math.isnan(mapped[1])
        assert mapped[2] == 200.0
        assert math.isnan(mapped[3])
        assert mapped[4] == 300.0

        restored = sentinel.replace_nans(mapped)
        assert restored[1] == -999.25
        assert restored[3] == -999.25

    def test_parse_ascii_line(self) -> None:
        """Parses whitespace-separated data line with automatic NaN mapping."""
        sentinel = NullSentinel.default_cwls()
        line = "  1670.000   2550.000  -999.2500   0.250  "
        vals = sentinel.parse_line(line)
        assert len(vals) == 4
        assert vals[0] == 1670.0
        assert vals[1] == 2550.0
        assert math.isnan(vals[2])
        assert vals[3] == 0.25

    def test_parse_ascii_line_empty_and_comment(self) -> None:
        """Empty and comment lines in data section return empty list."""
        sentinel = NullSentinel.default_cwls()
        assert sentinel.parse_line("") == []
        assert sentinel.parse_line("   ") == []
        assert sentinel.parse_line("# Comment line inside data") == []

    def test_parse_ascii_line_error(self) -> None:
        """Invalid non-numeric token raises ValueError."""
        sentinel = NullSentinel.default_cwls()
        with pytest.raises(ValueError, match="Invalid numerical token"):
            sentinel.parse_line("1670.0 BAD_TOKEN 2550.0")


class TestDynamicDetection:
    """Tests for dynamically detecting NULL sentinel from header items and lines."""

    def test_detect_from_header_item(self) -> None:
        """Constructs NullSentinel from parsed HeaderItem."""
        line = "NULL . -9999.0000 : NULL VALUE"
        item = HeaderItem.parse_line(line)
        assert item.is_null_record

        sentinel = NullSentinel.from_item(item)
        assert sentinel is not None
        assert sentinel.value == -9999.0
        assert sentinel.raw_sentinel == "-9999.0000"
        assert sentinel.is_null(-9999.0)

        # Non-null item returns None
        item_strt = HeaderItem.parse_line("STRT.M 100 : START")
        assert not item_strt.is_null_record
        assert NullSentinel.from_item(item_strt) is None

    def test_detect_from_lines(self) -> None:
        """Scans list of header lines and detects the NULL record."""
        lines = [
            "# Well block",
            "STRT .M 100.0 : START",
            "STOP .M 200.0 : STOP",
            "NULL . -9999.25 : NULL VALUE",
            "WELL . TEST WELL : WELL",
        ]
        detected = NullSentinel.detect(lines)
        assert detected is not None
        assert detected.value == -9999.25
        assert detected.raw_sentinel == "-9999.25"


class TestStateMachineIntegration:
    """Tests dynamic NULL detection in SectionStateMachine and parse_data_line."""

    def test_state_machine_dynamic_detection(self) -> None:
        """StateMachine dynamically updates NULL sentinel when encountering NULL in ~W."""
        sm = SectionStateMachine(strict=True)
        assert sm.null_value == -999.25
        assert not sm.is_null_detected

        # Step through sections
        sm.feed_line("~VERSION INFORMATION")
        sm.feed_line("VERS. 2.0 : CWLS LAS")
        sm.feed_line("WRAP. NO : ONE LINE PER DEPTH")

        sm.feed_line("~WELL INFORMATION")
        sm.feed_line("STRT.M 100.0 : START")
        assert not sm.is_null_detected

        # Feed dynamic NULL record
        sm.feed_line("NULL. -9999.0 : NULL VALUE")
        assert sm.is_null_detected
        assert sm.null_value == -9999.0
        assert sm.null_sentinel.value == -9999.0
        assert sm.null_sentinel.is_null(-9999.0)

        # Feed remaining sections to reach ~A
        sm.feed_line("~CURVE INFORMATION")
        sm.feed_line("DEPT.M : DEPTH")
        sm.feed_line("RHOB.K/M3 : DENSITY")

        sm.feed_line("~ASCII LOG DATA")
        # In ~A, parse_data_line uses the dynamically detected -9999.0 sentinel!
        data = sm.parse_data_line("100.0 -9999.0")
        assert data[0] == 100.0
        assert math.isnan(data[1])

        # Resetting restores CWLS default
        sm.reset()
        assert sm.null_value == -999.25
        assert not sm.is_null_detected

    def test_state_machine_data_line_strict_enforcement(self) -> None:
        """Strict mode rejects parse_data_line when outside the ~A section."""
        sm = SectionStateMachine(strict=True)
        sm.feed_line("~VERSION")
        with pytest.raises(LASSectionError, match="not currently in ASCII data section"):
            sm.parse_data_line("100.0 200.0")


class TestHeaderItemIntegration:
    """Tests HeaderItem null methods."""

    def test_header_item_null_methods(self) -> None:
        """Tests is_null and as_float_mapped on HeaderItem."""
        item_null = HeaderItem("NULL", "", "-999.25", "NULL VALUE")
        assert item_null.is_null_record
        assert item_null.is_null()
        assert math.isnan(item_null.as_float_mapped())  # type: ignore[arg-type]

        item_strt = HeaderItem("STRT", "M", "1670.0", "START")
        assert not item_strt.is_null_record
        assert not item_strt.is_null()
        assert item_strt.as_float_mapped() == 1670.0

        # Custom sentinel
        sentinel_9999 = NullSentinel(-9999.0)
        item_9999 = HeaderItem("PARAM", "", "-9999.0", "PARAM")
        assert item_9999.is_null(sentinel_9999)
        assert math.isnan(item_9999.as_float_mapped(sentinel_9999))  # type: ignore[arg-type]


class TestStandaloneHelpers:
    """Tests standalone module-level functions."""

    def test_standalone_functions(self) -> None:
        """Tests map_to_nan, map_from_nan, and is_null top-level helpers."""
        assert is_null(-999.25)
        assert is_null(float("nan"))
        assert not is_null(1670.0)

        assert math.isnan(map_to_nan(-999.25))
        assert map_to_nan(1670.0) == 1670.0

        assert map_from_nan(float("nan")) == -999.25
        assert map_from_nan(1670.0) == 1670.0

        # Custom sentinel argument
        assert is_null(-9999.0, sentinel=-9999.0)
        assert math.isnan(map_to_nan(-9999.0, sentinel=-9999.0))
        assert map_from_nan(float("nan"), sentinel=-9999.0) == -9999.0
