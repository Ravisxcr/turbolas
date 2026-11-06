"""TurboLAS: High-performance LAS (Log ASCII Standard) well log toolkit."""

from __future__ import annotations

try:
    from turbolas._turbolas import (
        DecodedBuffer,
        Encoding,
        HeaderItem,
        HeaderParser,
        LASHeaderError,
        LASSectionError,
        NullPolicy,
        NullSentinel,
        PyLineAction,
        SectionDelimiter,
        SectionStateMachine,
        SectionType,
        State,
        ValidationMode,
        cp1252_to_char,
        decode,
        decode_buffer,
        decode_cp1252,
        detect_encoding,
        is_null,
        is_valid_utf8,
        map_from_nan,
        map_to_nan,
        parse_header_line,
    )
    from turbolas._turbolas import (
        __version__ as _rust_version,
    )

    __version__: str = _rust_version
except ImportError:
    __version__ = "0.1.0-dev"

__all__ = [
    "DecodedBuffer",
    "Encoding",
    "HeaderItem",
    "HeaderParser",
    "LASHeaderError",
    "LASSectionError",
    "NullPolicy",
    "NullSentinel",
    "PyLineAction",
    "SectionDelimiter",
    "SectionStateMachine",
    "SectionType",
    "State",
    "ValidationMode",
    "__version__",
    "cp1252_to_char",
    "decode",
    "decode_buffer",
    "decode_cp1252",
    "detect_encoding",
    "is_null",
    "is_valid_utf8",
    "map_from_nan",
    "map_to_nan",
    "parse_header_line",
]
