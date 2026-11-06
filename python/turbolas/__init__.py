"""TurboLAS: High-performance LAS (Log ASCII Standard) well log toolkit."""

from __future__ import annotations

try:
    from turbolas._turbolas import (
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
        is_null,
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
    "is_null",
    "map_from_nan",
    "map_to_nan",
    "parse_header_line",
]
