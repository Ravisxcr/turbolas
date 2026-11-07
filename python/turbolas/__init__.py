"""TurboLAS: High-performance LAS (Log ASCII Standard) well log toolkit."""

from __future__ import annotations

import os
from typing import Any

try:
    from turbolas._turbolas import (
        ColumnarBuffer,
        ColumnarParser,
        DecodedBuffer,
        Encoding,
        HeaderItem,
        HeaderParser,
        LasCurve,
        LASDataError,
        LASHeaderError,
        LASSectionError,
        LasWell,
        NullPolicy,
        NullSentinel,
        OutputFormat,
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
    from turbolas._turbolas import (
        read_las as _rust_read_las,
    )

    __version__: str = _rust_version
except ImportError:
    __version__ = "0.1.0-dev"


def read_las(
    source: str | bytes | os.PathLike[str] | Any,
    validation_mode: ValidationMode | None = None,
    null_policy: NullPolicy | None = None,
    output_format: str = "numpy",
) -> LasWell:
    """Reads and parses a CWLS LAS file into a `LasWell` instance.

    Provides ergonomic dictionary-like curve lookups (e.g. `well["GR"]`),
    metadata accessors, and zero-copy export to NumPy, PyArrow, and Polars.

    Args:
        source: File path, raw LAS text string, bytes, or file-like object with `.read()`.
        validation_mode: Validation mode (`ValidationMode.Strict` or `ValidationMode.Permissive`).
        null_policy: Policy for handling null sentinel values (`NullPolicy.Strict`, etc.).
        output_format: Target format for `well[mnemonic]` lookups
            ('numpy', 'arrow', 'bytes', 'list').

    Returns:
        A `LasWell` instance with complete curve data and header metadata.
    """
    if hasattr(source, "read"):
        content = source.read()
        return read_las(
            content,
            validation_mode=validation_mode,
            null_policy=null_policy,
            output_format=output_format,
        )
    if isinstance(source, os.PathLike):
        source = str(source)
    return _rust_read_las(
        source,
        validation_mode=validation_mode,
        null_policy=null_policy,
        output_format=output_format,
    )


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
    "LasCurve",
    "LasWell",
    "NullPolicy",
    "NullSentinel",
    "OutputFormat",
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
    "read_las",
    "unwrap_and_parse",
]
