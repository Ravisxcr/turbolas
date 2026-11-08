//! Zero-copy encoding detection with SIMD-accelerated UTF-8 validation and automatic Windows-1252 fallback.
//!
//! # Architecture
//!
//! CWLS LAS (Log ASCII Standard) files are historically defined as 7-bit ASCII,
//! but modern files are frequently encoded in UTF-8 (with or without a Byte Order Mark).
//! Conversely, legacy petrophysical software written for Microsoft Windows frequently
//! emitted files encoded in Windows-1252 (CP1252 / ANSI), embedding degree symbols (`°` / `0xB0`),
//! micro signs (`µ` / `0xB5`), curly quotes, em-dashes, and accented letters directly as
//! 8-bit bytes in headers and parameter remarks.
//!
//! To achieve maximum throughput without intermediate allocations:
//! 1. Incoming raw byte slices (`&[u8]`) from memory-mapped files, byte buffers, or network streams
//!    are first checked for UTF-8 Byte Order Marks (`0xEF, 0xBB, 0xBF`).
//! 2. Fast vectorized ASCII verification (`is_ascii`) checks whether the slice is pure 7-bit ASCII.
//! 3. If non-ASCII bytes are present, vectorized SIMD UTF-8 validation (`simdutf8`) inspects
//!    the buffer in-place at gigabytes per second. If valid UTF-8, the slice is immediately
//!    cast into a borrowed string view (`Cow::Borrowed(&str)`) without heap allocation.
//! 4. If validation fails due to invalid multibyte sequences (e.g. standalone `0x80..=0xFF`),
//!    the pipeline intercepts the error without throwing an unhandled exception, falls back to
//!    decoding the buffer through a single-byte Windows-1252 mapping table (which maps every
//!    8-bit pattern `0x80..=0xFF` to valid Unicode scalar values), and only allocates a transformed
//!    string (`Cow::Owned(String)`) when conversion or replacement is strictly unavoidable.

use pyo3::prelude::*;
use std::borrow::Cow;
use std::fmt;
use std::ops::Deref;

/// Complete Windows-1252 (CP1252) byte-to-char mapping table for all 256 bytes (`0x00..=0xFF`).
///
/// Maps every 8-bit pattern to a valid Unicode scalar value:
/// - `0x00..=0x7F`: Standard 7-bit ASCII (identical to Unicode `U+0000..=U+007F`)
/// - `0x80..=0x9F`: Windows-1252 specific extensions (Euro, smart quotes, dashes, ligatures, etc.)
/// - `0xA0..=0xFF`: ISO-8859-1 equivalent (degree, micro, superscripts, accented letters, etc.)
pub const CP1252_MAP: [char; 256] = [
    // 0x00 - 0x0F (ASCII Control Characters)
    '\u{0000}', '\u{0001}', '\u{0002}', '\u{0003}', '\u{0004}', '\u{0005}', '\u{0006}', '\u{0007}',
    '\u{0008}', '\u{0009}', '\u{000A}', '\u{000B}', '\u{000C}', '\u{000D}', '\u{000E}', '\u{000F}',
    // 0x10 - 0x1F (ASCII Control Characters)
    '\u{0010}', '\u{0011}', '\u{0012}', '\u{0013}', '\u{0014}', '\u{0015}', '\u{0016}', '\u{0017}',
    '\u{0018}', '\u{0019}', '\u{001A}', '\u{001B}', '\u{001C}', '\u{001D}', '\u{001E}', '\u{001F}',
    // 0x20 - 0x2F (ASCII Printable)
    ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
    // 0x30 - 0x3F (Digits and punctuation)
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
    // 0x40 - 0x4F (Uppercase A-O)
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    // 0x50 - 0x5F (Uppercase P-Z, brackets)
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    // 0x60 - 0x6F (Lowercase a-o)
    '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    // 0x70 - 0x7F (Lowercase p-z, braces, DEL)
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '\u{007F}',
    // 0x80 - 0x8F (Windows-1252 extensions)
    '\u{20AC}', // 0x80: Euro Sign (€)
    '\u{0081}', // 0x81: Undefined in CP1252, mapped to U+0081
    '\u{201A}', // 0x82: Single Low-9 Quotation Mark (‚)
    '\u{0192}', // 0x83: Latin Small Letter F With Hook (ƒ)
    '\u{201E}', // 0x84: Double Low-9 Quotation Mark („)
    '\u{2026}', // 0x85: Horizontal Ellipsis (…)
    '\u{2020}', // 0x86: Dagger (†)
    '\u{2021}', // 0x87: Double Dagger (‡)
    '\u{02C6}', // 0x88: Modifier Letter Circumflex Accent (ˆ)
    '\u{2030}', // 0x89: Per Mille Sign (‰)
    '\u{0160}', // 0x8A: Latin Capital Letter S With Caron (Š)
    '\u{2039}', // 0x8B: Single Left-Pointing Angle Quotation Mark (‹)
    '\u{0152}', // 0x8C: Latin Capital Ligature OE (Œ)
    '\u{008D}', // 0x8D: Undefined in CP1252, mapped to U+008D
    '\u{017D}', // 0x8E: Latin Capital Letter Z With Caron (Ž)
    '\u{008F}', // 0x8F: Undefined in CP1252, mapped to U+008F
    // 0x90 - 0x9F (Windows-1252 extensions)
    '\u{0090}', // 0x90: Undefined in CP1252, mapped to U+0090
    '\u{2018}', // 0x91: Left Single Quotation Mark (‘)
    '\u{2019}', // 0x92: Right Single Quotation Mark (’)
    '\u{201C}', // 0x93: Left Double Quotation Mark (“)
    '\u{201D}', // 0x94: Right Double Quotation Mark (”)
    '\u{2022}', // 0x95: Bullet (•)
    '\u{2013}', // 0x96: En Dash (–)
    '\u{2014}', // 0x97: Em Dash (—)
    '\u{02DC}', // 0x98: Small Tilde (˜)
    '\u{2122}', // 0x99: Trade Mark Sign (™)
    '\u{0161}', // 0x9A: Latin Small Letter S With Caron (š)
    '\u{203A}', // 0x9B: Single Right-Pointing Angle Quotation Mark (›)
    '\u{0153}', // 0x9C: Latin Small Ligature OE (œ)
    '\u{009D}', // 0x9D: Undefined in CP1252, mapped to U+009D
    '\u{017E}', // 0x9E: Latin Small Letter Z With Caron (ž)
    '\u{0178}', // 0x9F: Latin Capital Letter Y With Diaeresis (Ÿ)
    // 0xA0 - 0xAF (ISO-8859-1 Symbols and Punctuation)
    '\u{00A0}', // 0xA0: No-Break Space
    '\u{00A1}', // 0xA1: Inverted Exclamation Mark (¡)
    '\u{00A2}', // 0xA2: Cent Sign (¢)
    '\u{00A3}', // 0xA3: Pound Sign (£)
    '\u{00A4}', // 0xA4: Currency Sign (¤)
    '\u{00A5}', // 0xA5: Yen Sign (¥)
    '\u{00A6}', // 0xA6: Broken Bar (¦)
    '\u{00A7}', // 0xA7: Section Sign (§)
    '\u{00A8}', // 0xA8: Diaeresis (¨)
    '\u{00A9}', // 0xA9: Copyright Sign (©)
    '\u{00AA}', // 0xAA: Feminine Ordinal Indicator (ª)
    '\u{00AB}', // 0xAB: Left-Pointing Double Angle Quotation Mark («)
    '\u{00AC}', // 0xAC: Not Sign (¬)
    '\u{00AD}', // 0xAD: Soft Hyphen
    '\u{00AE}', // 0xAE: Registered Sign (®)
    '\u{00AF}', // 0xAF: Macron (¯)
    // 0xB0 - 0xBF (ISO-8859-1 Units and Fractions)
    '\u{00B0}', // 0xB0: Degree Sign (°)
    '\u{00B1}', // 0xB1: Plus-Minus Sign (±)
    '\u{00B2}', // 0xB2: Superscript Two (²)
    '\u{00B3}', // 0xB3: Superscript Three (³)
    '\u{00B4}', // 0xB4: Acute Accent (´)
    '\u{00B5}', // 0xB5: Micro Sign (µ)
    '\u{00B6}', // 0xB6: Pilcrow Sign (¶)
    '\u{00B7}', // 0xB7: Middle Dot (·)
    '\u{00B8}', // 0xB8: Cedilla (¸)
    '\u{00B9}', // 0xB9: Superscript One (¹)
    '\u{00BA}', // 0xBA: Masculine Ordinal Indicator (º)
    '\u{00BB}', // 0xBB: Right-Pointing Double Angle Quotation Mark (»)
    '\u{00BC}', // 0xBC: Vulgar Fraction One Quarter (¼)
    '\u{00BD}', // 0xBD: Vulgar Fraction One Half (½)
    '\u{00BE}', // 0xBE: Vulgar Fraction Three Quarters (¾)
    '\u{00BF}', // 0xBF: Inverted Question Mark (¿)
    // 0xC0 - 0xCF (Latin Capital Letters with Diacritics)
    '\u{00C0}', '\u{00C1}', '\u{00C2}', '\u{00C3}', '\u{00C4}', '\u{00C5}', '\u{00C6}', '\u{00C7}',
    '\u{00C8}', '\u{00C9}', '\u{00CA}', '\u{00CB}', '\u{00CC}', '\u{00CD}', '\u{00CE}', '\u{00CF}',
    // 0xD0 - 0xDF (Latin Capital Letters & Multiplication Sign)
    '\u{00D0}', '\u{00D1}', '\u{00D2}', '\u{00D3}', '\u{00D4}', '\u{00D5}', '\u{00D6}', '\u{00D7}',
    '\u{00D8}', '\u{00D9}', '\u{00DA}', '\u{00DB}', '\u{00DC}', '\u{00DD}', '\u{00DE}', '\u{00DF}',
    // 0xE0 - 0xEF (Latin Small Letters with Diacritics)
    '\u{00E0}', '\u{00E1}', '\u{00E2}', '\u{00E3}', '\u{00E4}', '\u{00E5}', '\u{00E6}', '\u{00E7}',
    '\u{00E8}', '\u{00E9}', '\u{00EA}', '\u{00EB}', '\u{00EC}', '\u{00ED}', '\u{00EE}', '\u{00EF}',
    // 0xF0 - 0xFF (Latin Small Letters & Division Sign)
    '\u{00F0}', '\u{00F1}', '\u{00F2}', '\u{00F3}', '\u{00F4}', '\u{00F5}', '\u{00F6}', '\u{00F7}',
    '\u{00F8}', '\u{00F9}', '\u{00FA}', '\u{00FB}', '\u{00FC}', '\u{00FD}', '\u{00FE}', '\u{00FF}',
];

/// UTF-8 Byte Order Mark (`EF BB BF`).
pub const UTF8_BOM: &[u8; 3] = &[0xEF, 0xBB, 0xBF];

/// Supported character encodings detected during raw buffer analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pyclass(eq, eq_int)]
pub enum Encoding {
    /// Pure 7-bit ASCII (`0x00..=0x7F`), fully compliant with original CWLS LAS specification.
    Ascii,
    /// Standard UTF-8 without Byte Order Mark.
    Utf8,
    /// UTF-8 with Byte Order Mark (`0xEF, 0xBB, 0xBF`).
    Utf8Bom,
    /// Windows-1252 (CP1252 / ANSI) fallback encoding.
    Windows1252,
}

impl Encoding {
    /// Returns the canonical name of the encoding.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Ascii => "ASCII",
            Self::Utf8 => "UTF-8",
            Self::Utf8Bom => "UTF-8-SIG",
            Self::Windows1252 => "windows-1252",
        }
    }

    /// Returns `true` if the encoding is UTF-8 compatible (ASCII, UTF-8, or UTF-8-BOM).
    #[must_use]
    pub const fn is_utf8_compat(&self) -> bool {
        matches!(self, Self::Ascii | Self::Utf8 | Self::Utf8Bom)
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[pymethods]
impl Encoding {
    /// Canonical encoding name (e.g. `'ASCII'`, `'UTF-8'`, `'UTF-8-SIG'`, `'windows-1252'`).
    #[getter]
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.as_str()
    }

    /// Whether this encoding is UTF-8 (with or without BOM).
    #[getter]
    #[must_use]
    pub fn is_utf8(&self) -> bool {
        matches!(self, Self::Utf8 | Self::Utf8Bom)
    }

    /// Whether this encoding is pure 7-bit ASCII.
    #[getter]
    #[must_use]
    pub fn is_ascii(&self) -> bool {
        matches!(self, Self::Ascii)
    }

    /// Whether this encoding is Windows-1252.
    #[getter]
    #[must_use]
    pub fn is_windows1252(&self) -> bool {
        matches!(self, Self::Windows1252)
    }

    /// Whether this encoding starts with a UTF-8 Byte Order Mark.
    #[getter]
    #[must_use]
    pub fn has_bom(&self) -> bool {
        matches!(self, Self::Utf8Bom)
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __repr__(&self) -> String {
        format!(
            "Encoding.{}",
            match self {
                Self::Ascii => "Ascii",
                Self::Utf8 => "Utf8",
                Self::Utf8Bom => "Utf8Bom",
                Self::Windows1252 => "Windows1252",
            }
        )
    }

    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn __str__(&self) -> &'static str {
        self.as_str()
    }
}

/// Zero-allocation line iterator splitting on `\r\n`, `\n`, or standalone `\r`.
#[derive(Debug, Clone)]
pub struct LasLines<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> Iterator for LasLines<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();
        if self.pos >= bytes.len() {
            return None;
        }

        let start = self.pos;
        let mut i = start;
        while i < bytes.len() {
            if bytes[i] == b'\n' {
                let end = if i > start && bytes[i - 1] == b'\r' {
                    i - 1
                } else {
                    i
                };
                self.pos = i + 1;
                return Some(&self.text[start..end]);
            }
            if bytes[i] == b'\r' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                    let end = i;
                    self.pos = i + 2;
                    return Some(&self.text[start..end]);
                }
                // Standalone \r
                let end = i;
                self.pos = i + 1;
                return Some(&self.text[start..end]);
            }
            i += 1;
        }

        // Remainder of string without trailing newline
        self.pos = bytes.len();
        Some(&self.text[start..])
    }
}

/// Zero-copy decoded buffer holding either a borrowed string slice or an owned transformed `String`.
///
/// Implements `Deref<Target = str>` and `AsRef<str>` for seamless integration with downstream
/// slice parsers.
#[derive(Debug, Clone)]
pub struct DecodedBuffer<'a> {
    /// Decoded text view. Borrowed when valid UTF-8/ASCII; owned when CP1252 fallback was required.
    pub text: Cow<'a, str>,
    /// The encoding detected during analysis.
    pub encoding: Encoding,
    /// Whether a Byte Order Mark was present in the source bytes.
    pub has_bom: bool,
}

impl DecodedBuffer<'_> {
    /// Returns the text as a borrowed string slice.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Returns `true` if the text view was borrowed zero-copy without heap allocation.
    #[inline]
    #[must_use]
    pub fn is_borrowed(&self) -> bool {
        matches!(self.text, Cow::Borrowed(_))
    }

    /// Returns `true` if the text view required heap allocation (e.g. CP1252 conversion).
    #[inline]
    #[must_use]
    pub fn is_owned(&self) -> bool {
        matches!(self.text, Cow::Owned(_))
    }

    /// Converts the buffer into an owned `String`, avoiding re-allocation if already owned.
    #[must_use]
    pub fn into_owned(self) -> String {
        self.text.into_owned()
    }

    /// Returns the detected encoding.
    #[inline]
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Returns `true` if a BOM was stripped from the buffer.
    #[inline]
    #[must_use]
    pub const fn has_bom(&self) -> bool {
        self.has_bom
    }

    /// Returns the byte length of the decoded string.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Returns `true` if the decoded string is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Returns a zero-allocation iterator over lines, handling `\r\n`, `\n`, and standalone `\r`.
    #[must_use]
    pub fn lines(&self) -> LasLines<'_> {
        LasLines {
            text: &self.text,
            pos: 0,
        }
    }
}

impl Deref for DecodedBuffer<'_> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.text
    }
}

impl AsRef<str> for DecodedBuffer<'_> {
    #[inline]
    fn as_ref(&self) -> &str {
        &self.text
    }
}

impl fmt::Display for DecodedBuffer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Converts a single Windows-1252 byte (`0x00..=0xFF`) to its corresponding Unicode scalar value.
#[inline]
#[must_use]
pub const fn cp1252_byte_to_char(byte: u8) -> char {
    CP1252_MAP[byte as usize]
}

/// High-performance Windows-1252 (CP1252) decoder.
///
/// Uses chunked ASCII copying to batch-transfer contiguous runs of 7-bit ASCII bytes
/// (`< 0x80`) directly via `memcpy`, and maps non-ASCII bytes (`0x80..=0xFF`) through
/// the lookup table.
#[must_use]
pub fn decode_cp1252(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() + (bytes.len() >> 3).max(16));
    let mut idx = 0;

    while idx < bytes.len() {
        let start = idx;
        // Fast forward through contiguous ASCII bytes
        while idx < bytes.len() && bytes[idx] < 0x80 {
            idx += 1;
        }

        if idx > start {
            // Contiguous ASCII bytes are guaranteed valid UTF-8
            if let Ok(ascii_str) = std::str::from_utf8(&bytes[start..idx]) {
                out.push_str(ascii_str);
            }
        }

        if idx < bytes.len() {
            out.push(CP1252_MAP[bytes[idx] as usize]);
            idx += 1;
        }
    }

    out
}

/// Inspects a raw byte slice in-place with SIMD acceleration and returns a zero-copy
/// string view if valid UTF-8/ASCII, or falls back to Windows-1252 without raising an exception.
///
/// # Guarantees
/// - Zero allocations for any valid UTF-8 or ASCII byte buffer (returns `Cow::Borrowed(&str)`).
/// - Automatic UTF-8 Byte Order Mark (`0xEF, 0xBB, 0xBF`) detection and stripping without copying.
/// - Graceful Windows-1252 fallback for legacy 8-bit files (only allocates when conversion is strictly necessary).
#[must_use]
pub fn decode_buffer(bytes: &[u8]) -> DecodedBuffer<'_> {
    // 1. Check for UTF-8 Byte Order Mark (0xEF, 0xBB, 0xBF)
    if bytes.starts_with(UTF8_BOM) {
        let stripped = &bytes[3..];
        if let Ok(valid_str) = simdutf8::basic::from_utf8(stripped) {
            return DecodedBuffer {
                text: Cow::Borrowed(valid_str),
                encoding: Encoding::Utf8Bom,
                has_bom: true,
            };
        }
        // Fallback to CP1252 on the stripped slice
        return DecodedBuffer {
            text: Cow::Owned(decode_cp1252(stripped)),
            encoding: Encoding::Windows1252,
            has_bom: true,
        };
    }

    // 2. Vectorized pure 7-bit ASCII check (Rust stdlib uses SIMD internally)
    if bytes.is_ascii() {
        let ascii_str = std::str::from_utf8(bytes).unwrap_or_default();
        return DecodedBuffer {
            text: Cow::Borrowed(ascii_str),
            encoding: Encoding::Ascii,
            has_bom: false,
        };
    }

    // 3. SIMD-accelerated UTF-8 validation (AVX2 / SSE4.2 / NEON)
    match simdutf8::basic::from_utf8(bytes) {
        Ok(valid_str) => DecodedBuffer {
            text: Cow::Borrowed(valid_str),
            encoding: Encoding::Utf8,
            has_bom: false,
        },
        Err(_) => {
            // 4. Fall back to Windows-1252 single-byte decoding
            DecodedBuffer {
                text: Cow::Owned(decode_cp1252(bytes)),
                encoding: Encoding::Windows1252,
                has_bom: false,
            }
        }
    }
}

/// Detects the character encoding of a raw byte slice without heap allocation or full string transformation.
#[must_use]
pub fn detect_encoding(bytes: &[u8]) -> Encoding {
    if bytes.starts_with(UTF8_BOM) {
        let stripped = &bytes[3..];
        if simdutf8::basic::from_utf8(stripped).is_ok() {
            return Encoding::Utf8Bom;
        }
        return Encoding::Windows1252;
    }

    if bytes.is_ascii() {
        return Encoding::Ascii;
    }

    if simdutf8::basic::from_utf8(bytes).is_ok() {
        return Encoding::Utf8;
    }

    Encoding::Windows1252
}

/// Checks whether a raw byte slice is valid UTF-8 (handling BOM if present).
#[must_use]
pub fn is_valid_utf8(bytes: &[u8]) -> bool {
    let slice = if bytes.starts_with(UTF8_BOM) {
        &bytes[3..]
    } else {
        bytes
    };
    simdutf8::basic::from_utf8(slice).is_ok()
}

// ---------------------------------------------------------------------------
// PyO3 Python Bindings
// ---------------------------------------------------------------------------

/// Decoded buffer result exposed to Python with zero-copy metadata and text access.
#[pyclass(name = "DecodedBuffer")]
#[derive(Debug, Clone)]
pub struct PyDecodedBuffer {
    text: String,
    encoding: Encoding,
    is_zero_copy: bool,
    has_bom: bool,
    byte_length: usize,
}

#[pymethods]
impl PyDecodedBuffer {
    /// The decoded string content.
    #[getter]
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The detected encoding (`Encoding.Ascii`, `Encoding.Utf8`, `Encoding.Utf8Bom`, `Encoding.Windows1252`).
    #[getter]
    #[must_use]
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// Whether the Rust parser was able to borrow the buffer zero-copy without heap allocation.
    #[getter]
    #[must_use]
    pub fn is_zero_copy(&self) -> bool {
        self.is_zero_copy
    }

    /// Alias for `is_zero_copy`.
    #[getter]
    #[must_use]
    pub fn is_borrowed(&self) -> bool {
        self.is_zero_copy
    }

    /// Whether a Byte Order Mark was detected and stripped.
    #[getter]
    #[must_use]
    pub fn has_bom(&self) -> bool {
        self.has_bom
    }

    /// The length of the original byte buffer in bytes.
    #[getter]
    #[must_use]
    pub fn byte_length(&self) -> usize {
        self.byte_length
    }

    /// Returns a list of lines in the decoded text, handling `\r\n`, `\n`, and standalone `\r`.
    #[must_use]
    pub fn lines(&self) -> Vec<String> {
        LasLines {
            text: &self.text,
            pos: 0,
        }
        .map(String::from)
        .collect()
    }

    /// Number of lines in the decoded text.
    #[must_use]
    pub fn line_count(&self) -> usize {
        LasLines {
            text: &self.text,
            pos: 0,
        }
        .count()
    }

    fn __len__(&self) -> usize {
        self.text.len()
    }

    fn __str__(&self) -> &str {
        &self.text
    }

    fn __repr__(&self) -> String {
        format!(
            "DecodedBuffer(encoding={:?}, len={}, zero_copy={}, has_bom={})",
            self.encoding,
            self.text.len(),
            self.is_zero_copy,
            self.has_bom
        )
    }
}

/// Decodes a raw byte buffer with SIMD UTF-8 validation and automatic Windows-1252 fallback.
///
/// Returns a `DecodedBuffer` object with `.text`, `.encoding`, `.is_zero_copy`, and `.has_bom`.
#[pyfunction]
#[pyo3(name = "decode_buffer")]
#[must_use]
pub fn py_decode_buffer(data: &[u8]) -> PyDecodedBuffer {
    let decoded = decode_buffer(data);
    let is_zero_copy = decoded.is_borrowed();
    let encoding = decoded.encoding;
    let has_bom = decoded.has_bom;
    let byte_length = data.len();
    let text = decoded.into_owned();

    PyDecodedBuffer {
        text,
        encoding,
        is_zero_copy,
        has_bom,
        byte_length,
    }
}

/// Decodes raw bytes and returns a tuple `(text, encoding)`.
#[pyfunction]
#[pyo3(name = "decode")]
#[must_use]
pub fn py_decode(data: &[u8]) -> (String, Encoding) {
    let decoded = decode_buffer(data);
    let encoding = decoded.encoding;
    let text = decoded.into_owned();
    (text, encoding)
}

/// Detects the character encoding of raw bytes without full string conversion.
#[pyfunction]
#[pyo3(name = "detect_encoding")]
#[must_use]
pub fn py_detect_encoding(data: &[u8]) -> Encoding {
    detect_encoding(data)
}

/// Returns `True` if the byte slice is valid UTF-8 (handling BOM if present).
#[pyfunction]
#[pyo3(name = "is_valid_utf8")]
#[must_use]
pub fn py_is_valid_utf8(data: &[u8]) -> bool {
    is_valid_utf8(data)
}

/// Decodes raw bytes explicitly using the single-byte Windows-1252 table.
#[pyfunction]
#[pyo3(name = "decode_cp1252")]
#[must_use]
pub fn py_decode_cp1252(data: &[u8]) -> String {
    decode_cp1252(data)
}

/// Maps a single byte (`0..=255`) to its Windows-1252 Unicode character.
#[pyfunction]
#[pyo3(name = "cp1252_to_char")]
#[must_use]
pub fn py_cp1252_to_char(byte: u8) -> String {
    let c = cp1252_byte_to_char(byte);
    let mut s = String::with_capacity(4);
    s.push(c);
    s
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pure_ascii_zero_copy() {
        let raw = b"~VERSION INFORMATION\r\n VERS. 2.0: CWLS LOG ASCII STANDARD\r\n";
        let decoded = decode_buffer(raw);

        assert_eq!(decoded.encoding(), Encoding::Ascii);
        assert!(!decoded.has_bom());
        assert!(decoded.is_borrowed());
        assert!(!decoded.is_owned());
        assert_eq!(decoded.as_str(), std::str::from_utf8(raw).unwrap());
    }

    #[test]
    fn test_utf8_multibyte_zero_copy() {
        let text = "~WELL INFORMATION\n WELL. 100/01-01-001-01W4/0: Well with unicode \u{00B0}C and \u{00B5}s/ft\n";
        let raw = text.as_bytes();
        let decoded = decode_buffer(raw);

        assert_eq!(decoded.encoding(), Encoding::Utf8);
        assert!(!decoded.has_bom());
        assert!(decoded.is_borrowed());
        assert_eq!(decoded.as_str(), text);
    }

    #[test]
    fn test_utf8_bom_zero_copy() {
        let text = "~VERSION\n VERS. 2.0 : CWLS\n";
        let mut raw = UTF8_BOM.to_vec();
        raw.extend_from_slice(text.as_bytes());

        let decoded = decode_buffer(&raw);
        assert_eq!(decoded.encoding(), Encoding::Utf8Bom);
        assert!(decoded.has_bom());
        assert!(decoded.is_borrowed());
        assert_eq!(decoded.as_str(), text);
    }

    #[test]
    fn test_windows_1252_degree_symbol_fallback() {
        // Degree symbol in CP1252 is 0xB0 (in UTF-8 it is 0xC2, 0xB0)
        let raw = b"BHT .DEGC 120.5 \xB0C : BOTTOM HOLE TEMPERATURE";
        let decoded = decode_buffer(raw);

        assert_eq!(decoded.encoding(), Encoding::Windows1252);
        assert!(!decoded.has_bom());
        assert!(decoded.is_owned());
        assert!(!decoded.is_borrowed());
        assert_eq!(
            decoded.as_str(),
            "BHT .DEGC 120.5 \u{00B0}C : BOTTOM HOLE TEMPERATURE"
        );
    }

    #[test]
    fn test_windows_1252_micro_symbol_fallback() {
        // Micro sign in CP1252 is 0xB5 (in UTF-8 it is 0xC2, 0xB5)
        let raw = b"DT .US/F 65.5 \xB5s/ft : SONIC TRANSIT TIME";
        let decoded = decode_buffer(raw);

        assert_eq!(decoded.encoding(), Encoding::Windows1252);
        assert_eq!(
            decoded.as_str(),
            "DT .US/F 65.5 \u{00B5}s/ft : SONIC TRANSIT TIME"
        );
    }

    #[test]
    fn test_windows_1252_smart_quotes_and_dashes() {
        // 0x93 = left double quote (“), 0x94 = right double quote (”), 0x97 = em dash (—)
        let raw = b"COMP . \x93Schlumberger\x94 \x97 Well Services : COMPANY";
        let decoded = decode_buffer(raw);

        assert_eq!(decoded.encoding(), Encoding::Windows1252);
        assert_eq!(
            decoded.as_str(),
            "COMP . \u{201C}Schlumberger\u{201D} \u{2014} Well Services : COMPANY"
        );
    }

    #[test]
    fn test_windows_1252_euro_symbol() {
        // 0x80 = Euro sign (€ / U+20AC)
        let raw = b"COST . 1000 \x80 : LOGGING COST";
        let decoded = decode_buffer(raw);

        assert_eq!(decoded.encoding(), Encoding::Windows1252);
        assert_eq!(decoded.as_str(), "COST . 1000 \u{20AC} : LOGGING COST");
    }

    #[test]
    fn test_full_256_byte_sweep() {
        // Ensure every byte 0x00..=0xFF maps to a valid char without panic
        for b in 0..=255u8 {
            let c = cp1252_byte_to_char(b);
            assert!(c as u32 <= 0x0010_FFFF);
        }

        let all_bytes: Vec<u8> = (0..=255u8).collect();
        let decoded = decode_cp1252(&all_bytes);
        assert_eq!(decoded.chars().count(), 256);
    }

    #[test]
    fn test_detect_encoding() {
        assert_eq!(detect_encoding(b"PURE ASCII 123"), Encoding::Ascii);
        assert_eq!(
            detect_encoding("UTF-8 text with \u{00B0}C".as_bytes()),
            Encoding::Utf8
        );

        let mut bom_bytes = UTF8_BOM.to_vec();
        bom_bytes.extend_from_slice(b"BOM text");
        assert_eq!(detect_encoding(&bom_bytes), Encoding::Utf8Bom);

        assert_eq!(
            detect_encoding(b"Legacy text with \xB0C"),
            Encoding::Windows1252
        );
    }

    #[test]
    fn test_is_valid_utf8() {
        assert!(is_valid_utf8(b"Hello world"));
        assert!(is_valid_utf8("Degree \u{00B0}C".as_bytes()));
        assert!(!is_valid_utf8(b"Invalid \xB0C"));

        let mut bom = UTF8_BOM.to_vec();
        bom.extend_from_slice(b"Valid with BOM");
        assert!(is_valid_utf8(&bom));
    }

    #[test]
    fn test_empty_buffer() {
        let empty = b"";
        let decoded = decode_buffer(empty);
        assert_eq!(decoded.encoding(), Encoding::Ascii);
        assert!(decoded.is_borrowed());
        assert_eq!(decoded.as_str(), "");
        assert_eq!(decoded.len(), 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_lines_iteration() {
        let raw = b"Line 1\r\nLine 2\nLine 3\xB0";
        let decoded = decode_buffer(raw);
        let lines: Vec<&str> = decoded.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "Line 1");
        assert_eq!(lines[1], "Line 2");
        assert_eq!(lines[2], "Line 3\u{00B0}");
    }
}
