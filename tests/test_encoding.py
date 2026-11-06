"""Comprehensive test suite for zero-copy SIMD UTF-8 validation and Windows-1252 fallback."""

from __future__ import annotations

from turbolas import (
    DecodedBuffer,
    Encoding,
    cp1252_to_char,
    decode,
    decode_buffer,
    decode_cp1252,
    detect_encoding,
    is_valid_utf8,
)


class TestEncodingEnum:
    """Tests for Encoding enum variants, properties, and uppercase aliases."""

    def test_enum_variants(self) -> None:
        """Verifies standard enum variants exist and have correct properties."""
        assert Encoding.Ascii.name == "ASCII"
        assert Encoding.Ascii.is_ascii
        assert not Encoding.Ascii.is_utf8
        assert not Encoding.Ascii.is_windows1252
        assert not Encoding.Ascii.has_bom

        assert Encoding.Utf8.name == "UTF-8"
        assert Encoding.Utf8.is_utf8
        assert not Encoding.Utf8.is_ascii
        assert not Encoding.Utf8.is_windows1252
        assert not Encoding.Utf8.has_bom

        assert Encoding.Utf8Bom.name == "UTF-8-SIG"
        assert Encoding.Utf8Bom.is_utf8
        assert Encoding.Utf8Bom.has_bom
        assert not Encoding.Utf8Bom.is_ascii
        assert not Encoding.Utf8Bom.is_windows1252

        assert Encoding.Windows1252.name == "windows-1252"
        assert Encoding.Windows1252.is_windows1252
        assert not Encoding.Windows1252.is_utf8
        assert not Encoding.Windows1252.is_ascii
        assert not Encoding.Windows1252.has_bom

    def test_uppercase_aliases(self) -> None:
        """Verifies uppercase aliases for all enum variants."""
        assert Encoding.ASCII == Encoding.Ascii
        assert Encoding.UTF8 == Encoding.Utf8
        assert Encoding.UTF_8 == Encoding.Utf8
        assert Encoding.UTF8_BOM == Encoding.Utf8Bom
        assert Encoding.UTF8BOM == Encoding.Utf8Bom
        assert Encoding.WINDOWS_1252 == Encoding.Windows1252
        assert Encoding.WINDOWS1252 == Encoding.Windows1252
        assert Encoding.CP1252 == Encoding.Windows1252

    def test_string_representations(self) -> None:
        """Verifies str and repr formatting of Encoding enum."""
        assert str(Encoding.Ascii) == "ASCII"
        assert str(Encoding.Utf8) == "UTF-8"
        assert str(Encoding.Utf8Bom) == "UTF-8-SIG"
        assert str(Encoding.Windows1252) == "windows-1252"

        assert repr(Encoding.Ascii) == "Encoding.Ascii"
        assert repr(Encoding.Utf8) == "Encoding.Utf8"
        assert repr(Encoding.Utf8Bom) == "Encoding.Utf8Bom"
        assert repr(Encoding.Windows1252) == "Encoding.Windows1252"


class TestZeroCopyUtf8Validation:
    """Tests for zero-copy SIMD UTF-8 validation against pure ASCII and valid UTF-8 buffers."""

    def test_pure_ascii_zero_copy(self) -> None:
        """Pure 7-bit ASCII buffers are validated zero-copy without heap allocation."""
        raw = b"~VERSION INFORMATION\r\n VERS. 2.0: CWLS LOG ASCII STANDARD\r\n"
        buf = decode_buffer(raw)
        assert isinstance(buf, DecodedBuffer)

        assert buf.encoding == Encoding.Ascii
        assert buf.is_zero_copy
        assert buf.is_borrowed
        assert not buf.has_bom
        assert buf.byte_length == len(raw)
        assert buf.text == raw.decode("ascii")

    def test_multibyte_utf8_zero_copy(self) -> None:
        """Valid multibyte UTF-8 buffers are validated zero-copy with SIMD acceleration."""
        original = (
            "~WELL INFORMATION\n"
            " WELL. 100/01-01-001-01W4/0: Well with \u00b0C and \u00b5s/ft\n"
            " COMP. Petr\u00f3leo Brasileiro S.A. : COMPANY\n"
        )
        raw = original.encode("utf-8")
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Utf8
        assert buf.is_zero_copy
        assert buf.is_borrowed
        assert not buf.has_bom
        assert buf.text == original

    def test_utf8_bom_zero_copy(self) -> None:
        """UTF-8 buffers with BOM (EF BB BF) strip the BOM zero-copy."""
        text = "~VERSION\n VERS. 2.0 : CWLS\n"
        raw = b"\xef\xbb\xbf" + text.encode("utf-8")
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Utf8Bom
        assert buf.is_zero_copy
        assert buf.is_borrowed
        assert buf.has_bom
        assert buf.text == text
        assert not buf.text.startswith("\ufeff")

    def test_large_ascii_buffer_performance(self) -> None:
        """Large ASCII log data streams maintain zero-copy validation."""
        line = b" 1670.0000 -999.2500 0.1234 50.0000 120.4500\r\n"
        raw = b"~A DEPTH RHOB NPHI DT GR\r\n" + (line * 10_000)
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Ascii
        assert buf.is_zero_copy
        assert buf.byte_length == len(raw)
        assert buf.line_count() == 10_001


class TestWindows1252Fallback:
    """Tests for automatic Windows-1252 (CP1252) fallback on invalid multibyte sequences."""

    def test_degree_symbol_fallback(self) -> None:
        """Legacy 0xB0 byte decodes to degree symbol (\u00b0)."""
        raw = b"BHT .DEGC 120.5 \xb0C : BOTTOM HOLE TEMPERATURE"
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert not buf.is_zero_copy
        assert not buf.has_bom
        assert buf.text == "BHT .DEGC 120.5 \u00b0C : BOTTOM HOLE TEMPERATURE"

    def test_micro_symbol_fallback(self) -> None:
        """Legacy 0xB5 byte decodes to micro symbol (\u00b5)."""
        raw = b"DT .US/F 65.5 \xb5s/ft : SONIC TRANSIT TIME"
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert not buf.is_zero_copy
        assert buf.text == "DT .US/F 65.5 \u00b5s/ft : SONIC TRANSIT TIME"

    def test_smart_quotes_and_em_dash(self) -> None:
        """Windows-1252 smart quotes (0x93, 0x94) and em-dash (0x97) decode cleanly."""
        raw = b"COMP . \x93Schlumberger\x94 \x97 Wireline : COMPANY"
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert not buf.is_zero_copy
        assert buf.text == "COMP . \u201cSchlumberger\u201d \u2014 Wireline : COMPANY"

    def test_euro_symbol(self) -> None:
        """Euro symbol (0x80) decodes to \u20ac."""
        raw = b"FEE . 2500 \x80 : SERVICE CHARGE"
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert buf.text == "FEE . 2500 \u20ac : SERVICE CHARGE"

    def test_copyright_and_registered(self) -> None:
        """Copyright (0xA9) and Registered (0xAE) decode cleanly."""
        raw = b"PROD . UltraSonic\xae Logging System\xa9 : TOOL"
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert buf.text == "PROD . UltraSonic\u00ae Logging System\u00a9 : TOOL"

    def test_unassigned_cp1252_bytes_fallback(self) -> None:
        """Undefined CP1252 bytes (0x81, 0x8D, etc.) map to valid Unicode without error."""
        raw = b"\x81 \x8d \x8f \x90 \x9d"
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert len(buf.text) == 9
        assert buf.text == "\u0081 \u008d \u008f \u0090 \u009d"

    def test_full_256_byte_sweep(self) -> None:
        """All 256 possible 8-bit bytes map to valid Unicode characters."""
        raw = bytes(range(256))
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert len(buf.text) == 256
        for i, ch in enumerate(buf.text):
            expected = cp1252_to_char(i)
            assert ch == expected


class TestDecodedBufferMethods:
    """Tests for DecodedBuffer class methods, protocol implementation, and lines helper."""

    def test_buffer_metadata_and_dunder_methods(self) -> None:
        """Verifies len, str, repr, and getters on DecodedBuffer."""
        raw = b"Line 1\r\nLine 2\r\nLine 3"
        buf = decode_buffer(raw)

        assert len(buf) == len(raw)
        assert str(buf) == buf.text
        assert "DecodedBuffer" in repr(buf)
        assert "Ascii" in repr(buf)
        assert buf.byte_length == len(raw)

    def test_lines_and_line_count(self) -> None:
        """DecodedBuffer.lines() returns lines correctly splitting on CR/LF."""
        raw = b"Line 1\r\nLine 2\nLine 3\rLine 4"
        buf = decode_buffer(raw)

        assert buf.line_count() == 4
        lines = buf.lines()
        assert lines == ["Line 1", "Line 2", "Line 3", "Line 4"]


class TestStandaloneFunctions:
    """Tests for standalone module-level encoding helper functions."""

    def test_decode_helper_tuple(self) -> None:
        """turbolas.decode() returns (text, encoding) tuple."""
        text, enc = decode(b"~VERSION\n")
        assert text == "~VERSION\n"
        assert enc == Encoding.Ascii

        text_cp, enc_cp = decode(b"DEGC . \xb0C : UNIT")
        assert "\u00b0C" in text_cp
        assert enc_cp == Encoding.Windows1252

    def test_detect_encoding(self) -> None:
        """detect_encoding correctly identifies encoding without full string conversion."""
        assert detect_encoding(b"Hello world") == Encoding.Ascii
        assert detect_encoding("Well \u00b0C".encode("utf-8")) == Encoding.Utf8
        assert detect_encoding(b"\xef\xbb\xbfBOM") == Encoding.Utf8Bom
        assert detect_encoding(b"Legacy \xb0C") == Encoding.Windows1252

    def test_is_valid_utf8(self) -> None:
        """is_valid_utf8 validates UTF-8 streams including BOM handling."""
        assert is_valid_utf8(b"Hello")
        assert is_valid_utf8("Unicode \u00b0C".encode("utf-8"))
        assert is_valid_utf8(b"\xef\xbb\xbfWith BOM")
        assert not is_valid_utf8(b"Legacy \xb0C")

    def test_decode_cp1252_direct(self) -> None:
        """decode_cp1252 decodes directly with CP1252 mapping table."""
        res = decode_cp1252(b"100 \xb0C and 50 \xb5s")
        assert res == "100 \u00b0C and 50 \u00b5s"

    def test_cp1252_to_char(self) -> None:
        """cp1252_to_char maps individual byte to character."""
        assert cp1252_to_char(0x41) == "A"
        assert cp1252_to_char(0x80) == "\u20ac"
        assert cp1252_to_char(0xB0) == "\u00b0"
        assert cp1252_to_char(0xB5) == "\u00b5"


class TestEdgeCases:
    """Tests for edge cases: empty buffers, truncated UTF-8 sequences, mixed content."""

    def test_empty_buffer(self) -> None:
        """Empty buffer produces empty DecodedBuffer with Ascii encoding."""
        buf = decode_buffer(b"")
        assert buf.encoding == Encoding.Ascii
        assert buf.is_zero_copy
        assert buf.text == ""
        assert len(buf) == 0
        assert buf.byte_length == 0
        assert buf.lines() == []
        assert buf.line_count() == 0

    def test_truncated_utf8_fallback(self) -> None:
        """Truncated UTF-8 multibyte sequence falls back to CP1252 without throwing."""
        # 0xC2 is a 2-byte UTF-8 leader, but followed by ASCII space 0x20 instead of continuation
        raw = b"Invalid leader \xc2 alone"
        buf = decode_buffer(raw)

        assert buf.encoding == Encoding.Windows1252
        assert not buf.is_zero_copy
        assert "\u00c2 alone" in buf.text

    def test_single_byte_buffers(self) -> None:
        """Single-byte buffers of all types are handled properly."""
        assert decode_buffer(b"A").encoding == Encoding.Ascii
        assert decode_buffer(b"A").is_zero_copy

        assert decode_buffer(b"\x80").encoding == Encoding.Windows1252
        assert decode_buffer(b"\x80").text == "\u20ac"

        assert decode_buffer(b"\xb0").encoding == Encoding.Windows1252
        assert decode_buffer(b"\xb0").text == "\u00b0"
