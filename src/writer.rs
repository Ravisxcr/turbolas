//! Fast, Rust-backed CWLS LAS file writer.
//!
//! Provides high-throughput buffered serialization of [`LasWell`] instances
//! into standard-compliant CWLS LAS 1.2, 2.0, and 3.0 formatted text files or streams.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::columnar::ColumnData;
use crate::well::LasWell;

/// Options configuring the CWLS LAS serialization.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Target LAS specification version (1.2, 2.0, or 3.0). Default is 2.0.
    pub version: f64,
    /// Whether to wrap data rows across multiple lines (`WRAP. YES`). Default is false.
    pub wrap: bool,
    /// Delimiter for LAS 3.0 data columns ("SPACE", "COMMA", "TAB"). Default is None.
    pub delimiter: Option<String>,
    /// Floating point display precision for numeric curves. Default is 4 decimal places.
    pub float_precision: usize,
    /// Null sentinel value to write when a float is NaN. Default is -999.25.
    pub null_value: f64,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            version: 2.0,
            wrap: false,
            delimiter: None,
            float_precision: 4,
            null_value: -999.25,
        }
    }
}

/// Serializes a [`LasWell`] instance to a file path.
///
/// Uses an internal 64 KB `BufWriter` to maximize I/O throughput.
///
/// # Errors
/// Returns an `io::Error` if file creation or writing fails.
pub fn write_las_file<P: AsRef<Path>>(
    well: &LasWell,
    path: P,
    options: &WriteOptions,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);
    write_las_stream(well, &mut writer, options)?;
    writer.flush()
}

/// Serializes a [`LasWell`] instance to a memory `String`.
///
/// # Errors
/// Returns an `io::Error` if formatting fails.
pub fn write_las_string(well: &LasWell, options: &WriteOptions) -> io::Result<String> {
    let mut buffer = Vec::with_capacity(64 * 1024);
    write_las_stream(well, &mut buffer, options)?;
    String::from_utf8(buffer).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
}

/// Serializes a [`LasWell`] instance to an arbitrary [`Write`] stream.
///
/// # Errors
/// Returns an `io::Error` if writing to the stream fails.
#[allow(clippy::too_many_lines)]
pub fn write_las_stream<W: Write>(
    well: &LasWell,
    writer: &mut W,
    options: &WriteOptions,
) -> io::Result<()> {
    let is_las3 = (options.version - 3.0).abs() < 0.05 || well.is_las3();
    let is_las12 = (options.version - 1.2).abs() < 0.05;

    let null_sentinel = well.null_value().unwrap_or(options.null_value);
    let well_dlm = well.delimiter();
    let dlm = options
        .delimiter
        .as_deref()
        .or(well_dlm.as_deref())
        .unwrap_or("SPACE")
        .to_ascii_uppercase();

    // 1. Write ~Version Section
    if is_las3 {
        writeln!(writer, "~Version Information")?;
        writeln!(
            writer,
            " VERS.                        3.0 : CWLS LOG ASCII STANDARD -VERSION 3.0"
        )?;
        let wrap_str = if options.wrap { "YES" } else { "NO" };
        writeln!(
            writer,
            " WRAP.                        {wrap_str:<4} : ONE LINE PER DEPTH STEP"
        )?;
        writeln!(
            writer,
            " DLM .                      {dlm:<5} : DELIMITING CHARACTER BETWEEN DATA COLUMNS"
        )?;
    } else if is_las12 {
        writeln!(writer, "~VERSION INFORMATION")?;
        writeln!(
            writer,
            " VERS.   1.2 : CWLS LOG ASCII STANDARD - VERSION 1.2"
        )?;
        let wrap_str = if options.wrap { "YES" } else { "NO" };
        writeln!(writer, " WRAP.   {wrap_str:<3} : ONE LINE PER DEPTH STEP")?;
    } else {
        writeln!(writer, "~VERSION INFORMATION")?;
        writeln!(
            writer,
            " VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0"
        )?;
        let wrap_str = if options.wrap { "YES" } else { "NO" };
        writeln!(writer, " WRAP.   {wrap_str:<3} : ONE LINE PER DEPTH STEP")?;
    }
    writeln!(writer)?;

    // 2. Write ~Well Section
    if is_las3 {
        writeln!(writer, "~Well Information")?;
        writeln!(
            writer,
            "#MNEM.UNIT             DATA                    DESCRIPTION"
        )?;
        writeln!(
            writer,
            "#----------------     -----------              -------------------------"
        )?;
    } else {
        writeln!(writer, "~WELL INFORMATION")?;
        writeln!(
            writer,
            "#MNEM.UNIT              DATA                   DESCRIPTION"
        )?;
        writeln!(
            writer,
            "#---------              ----                   -----------"
        )?;
    }

    // Write well items
    write_well_items(well, writer, null_sentinel, options.float_precision)?;
    writeln!(writer)?;

    // 3. Write ~Parameter Section (if present)
    let param_items = well.parameter_items();
    if !param_items.is_empty() {
        if is_las3 {
            writeln!(writer, "~Log_Parameter")?;
        } else {
            writeln!(writer, "~PARAMETER INFORMATION")?;
        }
        for item in param_items {
            writeln!(writer, "{}", item.format_line(Some(12), Some(6), Some(20)))?;
        }
        writeln!(writer)?;
    }

    // 4. Write ~Curve Section
    if is_las3 {
        writeln!(writer, "~Log_Definition")?;
        writeln!(
            writer,
            "#MNEM.UNIT             LOG CODES               CURVE DESCRIPTION"
        )?;
        writeln!(
            writer,
            "#----------------     -----------              -------------------------"
        )?;
    } else {
        writeln!(writer, "~CURVE INFORMATION")?;
        writeln!(
            writer,
            "#MNEM.UNIT             API CODE                CURVE DESCRIPTION"
        )?;
        writeln!(
            writer,
            "#---------             --------                -----------------"
        )?;
    }

    let curve_items = well.curve_items();
    for item in curve_items {
        writeln!(writer, "{}", item.format_line(Some(12), Some(6), Some(16)))?;
    }
    writeln!(writer)?;

    // 5. Write ~Other Section (if present)
    let other_lines = well.other_lines();
    if !other_lines.is_empty() {
        writeln!(writer, "~OTHER")?;
        for line in other_lines {
            writeln!(writer, "{line}")?;
        }
        writeln!(writer)?;
    }

    // 6. Write Data Section (~A or ~Log_Data)
    let delimiter_type = if is_las3 { dlm } else { "SPACE".to_string() };

    if is_las3 {
        writeln!(writer, "~Log_Data | Log_Definition")?;
    } else {
        writeln!(writer, "~A")?;
    }

    write_primary_data(
        well,
        writer,
        options.wrap,
        &delimiter_type,
        null_sentinel,
        options.float_precision,
    )?;

    // 7. If LAS 3.0, write additional sections (e.g. ~Core_Definition and ~Core_Data)
    if is_las3 {
        for sec_name in well.section_names() {
            let upper = sec_name.to_ascii_uppercase();
            if upper == "LOG" || upper == "ASCII" || upper == "CURVE" {
                continue;
            }

            // Write definition section
            if let Some(defs) = well.section_curves(&upper) {
                writeln!(writer)?;
                writeln!(writer, "~{sec_name}_Definition")?;
                for def in defs {
                    writeln!(writer, "{}", def.format_line(Some(12), Some(6), Some(16)))?;
                }
            }

            // Write data section
            if let Some(data_buf) = well.section_data_buffer(&upper) {
                writeln!(writer)?;
                writeln!(writer, "~{sec_name}_Data | {sec_name}_Definition")?;
                write_columnar_buffer_data(
                    &data_buf,
                    writer,
                    options.wrap,
                    &delimiter_type,
                    null_sentinel,
                    options.float_precision,
                )?;
            }
        }
    }

    Ok(())
}

#[allow(clippy::similar_names)]
fn write_well_items<W: Write>(
    well: &LasWell,
    writer: &mut W,
    null_sentinel: f64,
    precision: usize,
) -> io::Result<()> {
    let items = well.well_items();
    if items.is_empty() {
        // Fallback default well items
        let strt = well.start_depth().unwrap_or(0.0);
        let stop = well.stop_depth().unwrap_or(0.0);
        let step = well.step().unwrap_or(0.0);
        let well_name = well.well_name().unwrap_or_else(|| "UNKNOWN".to_string());
        let comp = well.company().unwrap_or_default();
        let fld = well.field().unwrap_or_default();
        let loc = well.location().unwrap_or_default();
        let srvc = well.service_company().unwrap_or_default();
        let date = well.date().unwrap_or_default();
        let uwi = well.uwi().unwrap_or_default();

        writeln!(writer, " STRT.M        {strt:>14.precision$} : START DEPTH")?;
        writeln!(writer, " STOP.M        {stop:>14.precision$} : STOP DEPTH")?;
        writeln!(writer, " STEP.M        {step:>14.precision$} : STEP VALUE")?;
        writeln!(
            writer,
            " NULL.         {null_sentinel:>14.precision$} : NULL VALUE"
        )?;
        writeln!(writer, " COMP.         {comp:<14} : COMPANY")?;
        writeln!(writer, " WELL.         {well_name:<14} : WELL")?;
        writeln!(writer, " FLD .         {fld:<14} : FIELD")?;
        writeln!(writer, " LOC .         {loc:<14} : LOCATION")?;
        writeln!(writer, " SRVC.         {srvc:<14} : SERVICE COMPANY")?;
        writeln!(writer, " DATE.         {date:<14} : DATE")?;
        writeln!(writer, " UWI .         {uwi:<14} : UNIQUE WELL ID")?;
    } else {
        for item in items {
            writeln!(writer, "{}", item.format_line(Some(12), Some(6), Some(20)))?;
        }
    }
    Ok(())
}

fn write_primary_data<W: Write>(
    well: &LasWell,
    writer: &mut W,
    wrap: bool,
    delimiter: &str,
    null_sentinel: f64,
    precision: usize,
) -> io::Result<()> {
    write_columnar_buffer_data(
        well.inner_data(),
        writer,
        wrap,
        delimiter,
        null_sentinel,
        precision,
    )
}

fn write_columnar_buffer_data<W: Write>(
    data: &crate::columnar::ColumnarBuffer,
    writer: &mut W,
    wrap: bool,
    delimiter: &str,
    null_sentinel: f64,
    precision: usize,
) -> io::Result<()> {
    let num_rows = data.num_rows();
    let num_cols = data.num_columns();
    if num_rows == 0 || num_cols == 0 {
        return Ok(());
    }

    let is_comma = delimiter == "COMMA" || delimiter == ",";
    let is_tab = delimiter == "TAB" || delimiter == "\t";

    let sep = if is_comma {
        ","
    } else if is_tab {
        "\t"
    } else {
        " "
    };

    let mut line_buf = String::with_capacity(num_cols * 16);

    for row_idx in 0..num_rows {
        line_buf.clear();

        for col_idx in 0..num_cols {
            if col_idx > 0 && !is_comma && !is_tab && !wrap {
                line_buf.push(' ');
            } else if col_idx > 0 && (is_comma || is_tab) {
                line_buf.push_str(sep);
            }

            match &data.column_data(col_idx) {
                Some(ColumnData::Float(col)) => {
                    let val = col[row_idx];
                    let out_val = if val.is_nan() { null_sentinel } else { val };
                    if is_comma || is_tab {
                        use std::fmt::Write;
                        let _ = write!(line_buf, "{out_val:.precision$}");
                    } else {
                        use std::fmt::Write;
                        let _ = write!(line_buf, "{out_val:>11.precision$}");
                    }
                }
                Some(ColumnData::Text(col)) => {
                    let text = &col[row_idx];
                    if text.contains(' ')
                        || text.contains(',')
                        || text.contains('\t')
                        || text.contains('"')
                    {
                        line_buf.push('"');
                        line_buf.push_str(text);
                        line_buf.push('"');
                    } else {
                        line_buf.push_str(text);
                    }
                }
                None => {}
            }

            // In wrapped mode (LAS 2.0 / 1.2), line 1 has depth, subsequent curves wrap
            if wrap
                && ((col_idx == 0 && num_cols > 1)
                    || (col_idx > 0 && col_idx < num_cols - 1 && (col_idx % 4 == 0)))
            {
                line_buf.push('\n');
            }
        }

        writeln!(writer, "{line_buf}")?;
    }

    Ok(())
}
