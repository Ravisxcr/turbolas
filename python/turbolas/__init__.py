"""TurboLAS: High-performance LAS (Log ASCII Standard) well log toolkit."""

from __future__ import annotations

try:
    from turbolas._turbolas import (
        ColumnarBuffer,
        ColumnarParser,
        DecodedBuffer,
        Encoding,
        HeaderItem,
        HeaderParser,
        LASDataError,
        LASHeaderError,
        LASSectionError,
        NullPolicy,
        NullSentinel,
        PyLineAction,
        SectionDelimiter,
        SectionStateMachine,
        SectionType,
        State,
        StreamingUnwrapper,
        UnwrappedRecord,
        ValidationMode,
        cp1252_to_char,
        decode,
        decode_buffer,
        decode_cp1252,
        detect_encoding,
        estimate_row_capacity,
        is_null,
        is_valid_utf8,
        map_from_nan,
        map_to_nan,
        parse_columnar,
        parse_header_line,
        unwrap_and_parse,
    )
    from turbolas._turbolas import (
        __version__ as _rust_version,
    )

    __version__: str = _rust_version
except ImportError:
    __version__ = "0.1.0-dev"

__all__ = [
    "ColumnarBuffer",
    "ColumnarParser",
    "DecodedBuffer",
    "Encoding",
    "HeaderItem",
    "HeaderParser",
    "LASDataError",
    "LASHeaderError",
    "LASSectionError",
    "NullPolicy",
    "NullSentinel",
    "PyLineAction",
    "SectionDelimiter",
    "SectionStateMachine",
    "SectionType",
    "State",
    "StreamingUnwrapper",
    "UnwrappedRecord",
    "ValidationMode",
    "__version__",
    "cp1252_to_char",
    "decode",
    "decode_buffer",
    "decode_cp1252",
    "detect_encoding",
    "estimate_row_capacity",
    "is_null",
    "is_valid_utf8",
    "map_from_nan",
    "map_to_nan",
    "parse_columnar",
    "parse_header_line",
    "unwrap_and_parse",
]
