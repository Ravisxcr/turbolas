//! End-to-end integration tests targeting canonical CWLS LAS 1.2 and 2.0 reference datasets.
//!
//! Validates parser compliance against real-world well log edge cases:
//! - Canonical CWLS 1.2 and 2.0 reference datasets
//! - Wrapped and unwrapped (`WRAP = YES/NO`) ASCII data sections
//! - Exact numerical parity between wrapped and unwrapped datasets
//! - Custom delimiter variants (spaces, tabs, mixed whitespace, CRLF)
//! - Mixed null sentinels and IEEE 754 quiet NaN mapping
//! - Zero-copy decoding integrity across UTF-8, BOM, and Windows-1252 (CP1252)
//! - Channel ordering and raw byte buffer consistency

#![allow(
    clippy::assert_is_empty,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::cast_precision_loss
)]

use crate::encoding::decode_buffer;
use crate::section::SectionType;
use crate::state_machine::ValidationMode;
use crate::well::{parse_las_bytes, OutputFormat};

const CWLS_12_CANONICAL: &str = include_str!("../tests/data/cwls_12_canonical.las");
const CWLS_20_CANONICAL_UNWRAPPED: &str =
    include_str!("../tests/data/cwls_20_canonical_unwrapped.las");
const CWLS_20_CANONICAL_WRAPPED: &str = include_str!("../tests/data/cwls_20_canonical_wrapped.las");
const CWLS_20_CUSTOM_SENTINEL: &str = include_str!("../tests/data/cwls_20_custom_sentinel.las");
const CWLS_20_TAB_DELIMITED: &[u8] = include_bytes!("../tests/data/cwls_20_tab_delimited_crlf.las");
const CWLS_20_CP1252: &[u8] = include_bytes!("../tests/data/cwls_20_cp1252_encoding.las");
const CWLS_20_DIVERSE_HEADERS: &str = include_str!("../tests/data/cwls_20_diverse_headers.las");

/// Helper to verify whether an `f64` is an IEEE 754 quiet NaN.
fn is_quiet_nan(val: f64) -> bool {
    if !val.is_nan() {
        return false;
    }
    let bits = val.to_bits();
    // In IEEE 754 float64, bit 51 (the most significant bit of the 52-bit fraction)
    // is the quiet/signaling NaN bit. If it is 1, the NaN is quiet.
    (bits & 0x0008_0000_0000_0000) != 0
}

#[test]
fn test_canonical_cwls_12_reference() {
    let well = parse_las_bytes(
        CWLS_12_CANONICAL.as_bytes(),
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse CWLS 1.2 reference dataset");

    // Specification version & wrap flag
    assert_eq!(well.version(), Some("1.2".to_string()));
    assert!(!well.is_wrapped());

    // Well identification metadata
    assert_eq!(well.uwi(), Some("100123401234W400".to_string()));
    assert_eq!(well.well_name(), Some("ANY ET AL 12-34-12-34".to_string()));
    assert_eq!(well.company(), Some("ANY OIL COMPANY INC.".to_string()));
    assert_eq!(well.field(), Some("EDAM".to_string()));
    assert_eq!(well.location(), Some("A12-34-12-34W4M".to_string()));
    assert_eq!(
        well.service_company(),
        Some("ANY LOGGING COMPANY INC.".to_string())
    );
    assert_eq!(well.date(), Some("25-DEC-1988".to_string()));

    // Depth step parameters (decreasing depth)
    assert_eq!(well.start_depth(), Some(1670.0));
    assert_eq!(well.stop_depth(), Some(1660.0));
    assert_eq!(well.step(), Some(-0.125));
    assert_eq!(well.null_value(), Some(-999.25));

    // Channel count & order
    assert_eq!(well.num_curves(), 8);
    assert_eq!(well.num_rows(), 5);
    let expected_channels = vec!["DEPT", "DT", "RHOB", "NPHI", "SFLU", "SFLA", "ILM", "ILD"];
    assert_eq!(well.curve_names(), expected_channels);

    // Verify curve metadata extractions (API codes in value field)
    let dt_curve = well.curve_items.get(1).expect("Missing DT curve item");
    assert_eq!(dt_curve.mnemonic(), "DT");
    assert_eq!(dt_curve.unit(), "US/M");
    assert_eq!(dt_curve.value(), "60 520 01 00");
    assert!(dt_curve.description().contains("SONIC TRANSIT TIME"));

    // Verify parameter section extractions
    assert_eq!(
        well.get_header_value(SectionType::Parameter, "BHT"),
        Some("35.5000")
    );
    assert_eq!(
        well.get_header_value(SectionType::Parameter, "BS"),
        Some("200.0000")
    );
    assert_eq!(
        well.get_header_value(SectionType::Parameter, "RMF"),
        Some("0.2160")
    );

    // Verify ~O remarks section
    assert!(!well.other().is_empty());
    assert!(well.other()[0].contains("stuck at 1665 meters"));

    // Ground truth numeric curve parity
    let dept = well.curve_data_by_name("DEPT").unwrap();
    assert_eq!(dept, &[1670.0, 1669.875, 1669.750, 1669.625, 1669.500]);

    let dt = well.curve_data_by_name("DT").unwrap();
    assert_eq!(dt[0], 123.45);
    assert_eq!(dt[1], 123.45);
    assert_eq!(dt[2], 124.10);
    assert!(dt[3].is_nan());
    assert!(is_quiet_nan(dt[3]));
    assert!(dt[4].is_nan());
    assert!(is_quiet_nan(dt[4]));

    let rhob = well.curve_data_by_name("RHOB").unwrap();
    assert_eq!(rhob[0], 2550.0);
    assert_eq!(rhob[1], 2550.0);
    assert_eq!(rhob[2], 2548.5);
    assert_eq!(rhob[3], 2545.0);
    assert!(rhob[4].is_nan());
    assert!(is_quiet_nan(rhob[4]));

    let ilm = well.curve_data_by_name("ILM").unwrap();
    assert_eq!(ilm, &[110.2, 110.2, 111.0, 112.5, 113.0]);

    let ild = well.curve_data_by_name("ILD").unwrap();
    assert_eq!(ild, &[105.6, 105.6, 106.2, 107.0, 108.4]);
}

#[test]
fn test_canonical_cwls_20_unwrapped() {
    let well = parse_las_bytes(
        CWLS_20_CANONICAL_UNWRAPPED.as_bytes(),
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse CWLS 2.0 unwrapped reference dataset");

    assert_eq!(well.version(), Some("2.0".to_string()));
    assert!(!well.is_wrapped());
    assert_eq!(well.num_curves(), 8);
    assert_eq!(well.num_rows(), 6);
    assert_eq!(well.start_depth(), Some(1670.0));
    assert_eq!(well.stop_depth(), Some(1669.375));
    assert_eq!(well.null_value(), Some(-999.25));

    let dept = well.curve_data_by_name("DEPT").unwrap();
    assert_eq!(
        dept,
        &[1670.0, 1669.875, 1669.750, 1669.625, 1669.500, 1669.375]
    );

    let dt = well.curve_data_by_name("DT").unwrap();
    assert_eq!(dt[0], 123.45);
    assert_eq!(dt[1], 123.45);
    assert_eq!(dt[2], 124.10);
    assert_eq!(dt[3], 125.00);
    assert_eq!(dt[4], 126.25);
    assert!(dt[5].is_nan());
    assert!(is_quiet_nan(dt[5]));
}

#[test]
fn test_canonical_cwls_20_wrapped_parity() {
    let unwrapped = parse_las_bytes(
        CWLS_20_CANONICAL_UNWRAPPED.as_bytes(),
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse unwrapped reference");

    let wrapped = parse_las_bytes(
        CWLS_20_CANONICAL_WRAPPED.as_bytes(),
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse wrapped reference");

    // Version flag differences
    assert!(!unwrapped.is_wrapped());
    assert!(wrapped.is_wrapped());

    // Both files must have identical curve counts, names, and rows
    assert_eq!(unwrapped.num_curves(), wrapped.num_curves());
    assert_eq!(unwrapped.num_rows(), wrapped.num_rows());
    assert_eq!(unwrapped.curve_names(), wrapped.curve_names());

    // Check exact numerical parity across every channel and every depth
    for (idx, name) in unwrapped.curve_names().iter().enumerate() {
        let u_col = unwrapped.curve_data(idx).unwrap();
        let w_col = wrapped.curve_data(idx).unwrap();
        assert_eq!(u_col.len(), w_col.len());

        for (row_idx, (&u_val, &w_val)) in u_col.iter().zip(w_col.iter()).enumerate() {
            if u_val.is_nan() {
                assert!(
                    w_val.is_nan(),
                    "Channel {name} at row {row_idx}: unwrapped is NaN but wrapped is {w_val}"
                );
                assert!(
                    is_quiet_nan(w_val),
                    "Channel {name} at row {row_idx}: wrapped NaN is not quiet NaN"
                );
            } else {
                assert_eq!(
                    u_val.to_bits(),
                    w_val.to_bits(),
                    "Channel {name} at row {row_idx}: bit mismatch (unwrapped={u_val}, wrapped={w_val})"
                );
            }
        }
    }
}

#[test]
fn test_custom_sentinel_and_quiet_nan() {
    let well = parse_las_bytes(
        CWLS_20_CUSTOM_SENTINEL.as_bytes(),
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse custom sentinel dataset");

    assert_eq!(well.null_value(), Some(-9999.0));
    assert_eq!(
        well.curve_names(),
        vec!["DEPTH", "GR", "NPHI", "CALI", "DTC"]
    );
    assert_eq!(well.num_rows(), 5);

    let gr = well.curve_data_by_name("GR").unwrap();
    assert_eq!(gr[0], 55.40);
    // Row 1: -9999.00 mapped to NaN
    assert!(gr[1].is_nan());
    assert!(is_quiet_nan(gr[1]));
    assert_eq!(gr[2], 58.20);
    // Row 3: -999.25 is NOT the declared sentinel (-9999.00), so it remains -999.25!
    assert_eq!(gr[3], -999.25);
    assert!(!gr[3].is_nan());
    assert_eq!(gr[4], 62.00);

    let nphi = well.curve_data_by_name("NPHI").unwrap();
    assert_eq!(nphi[0], 0.22);
    assert_eq!(nphi[1], 0.24);
    // Row 2: -9999.00 mapped to NaN
    assert!(nphi[2].is_nan());
    assert!(is_quiet_nan(nphi[2]));

    let cali = well.curve_data_by_name("CALI").unwrap();
    // Row 3: -9999.00 mapped to NaN
    assert!(cali[3].is_nan());
    assert!(is_quiet_nan(cali[3]));

    let dtc = well.curve_data_by_name("DTC").unwrap();
    // Row 3: text 'NaN' mapped to NaN
    assert!(dtc[3].is_nan());
    assert!(is_quiet_nan(dtc[3]));

    // Curve statistics ignoring NaNs
    let gr_curve = well.curve_items.get(1).unwrap();
    let rich_gr = well.curve_data_by_name(gr_curve.mnemonic()).unwrap();
    let non_nan_gr: Vec<f64> = rich_gr.iter().copied().filter(|v| !v.is_nan()).collect();
    assert_eq!(non_nan_gr.len(), 4);
}

#[test]
fn test_tab_delimited_and_crlf_log_data() {
    let well = parse_las_bytes(
        CWLS_20_TAB_DELIMITED,
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse tab-delimited CRLF dataset");

    assert_eq!(well.curve_names(), vec!["DEPT", "GR", "RHOB", "NPHI"]);
    assert_eq!(well.num_rows(), 5);
    assert_eq!(well.start_depth(), Some(2000.0));
    assert_eq!(well.stop_depth(), Some(2002.0));

    let dept = well.curve_data_by_name("DEPT").unwrap();
    assert_eq!(dept, &[2000.0, 2000.5, 2001.0, 2001.5, 2002.0]);

    let gr = well.curve_data_by_name("GR").unwrap();
    assert_eq!(gr[0], 45.5);
    assert_eq!(gr[1], 48.2);
    assert!(gr[2].is_nan()); // -999.25
    assert_eq!(gr[3], 52.1);
    assert_eq!(gr[4], 55.0);
}

#[test]
fn test_cp1252_encoding_fallback_and_metadata() {
    let decoded = decode_buffer(CWLS_20_CP1252);
    // Validation: CP1252 was detected and converted without exception
    assert!(!decoded.as_str().is_empty());

    let well = parse_las_bytes(
        CWLS_20_CP1252,
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse CP1252 encoded dataset");

    // Check special character extraction in metadata
    assert_eq!(
        well.well_name(),
        Some("“TROLL FIELD EXTENSION °1”".to_string())
    );
    assert_eq!(
        well.company(),
        Some("“TOTAL ENERGIES EP NORGE AS”".to_string())
    );

    // Check curve units with micro and degree symbols
    let dt_item = well.curve_items.get(2).expect("Missing DT curve");
    assert_eq!(dt_item.unit(), "µS/M");

    let temp_item = well.curve_items.get(3).expect("Missing TEMP curve");
    assert_eq!(temp_item.unit(), "°C");

    // Check parameter values and units
    let bht_item = well
        .get_header_item(SectionType::Parameter, "BHT")
        .expect("Missing BHT");
    assert_eq!(bht_item.unit(), "°C");
    assert_eq!(bht_item.value(), "78.5000");

    // Check ~O remarks containing copyright and euro signs
    let remarks = well.other();
    assert!(!remarks.is_empty());
    assert!(remarks[0].contains('©'));
    assert!(remarks[0].contains('€'));

    // Check numeric log values
    let dept = well.curve_data_by_name("DEPT").unwrap();
    assert_eq!(dept, &[3000.0, 3000.5, 3001.0, 3001.5]);

    let temp = well.curve_data_by_name("TEMP").unwrap();
    assert_eq!(temp, &[68.5, 69.0, 69.5, 70.0]);
}

#[test]
fn test_diverse_headers_and_version_quirks() {
    let well = parse_las_bytes(
        CWLS_20_DIVERSE_HEADERS.as_bytes(),
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .expect("Failed to parse diverse headers dataset");

    // Lowercase ~v and ~w sections parsed correctly
    assert_eq!(well.version(), Some("2.0".to_string()));
    assert_eq!(well.well_name(), Some("DIVERSE RIG #4".to_string()));
    assert_eq!(
        well.company(),
        Some("\"CHEVRON CANADA RESOURCES\"".to_string())
    );
    assert_eq!(well.start_depth(), Some(5000.0));
    assert_eq!(well.stop_depth(), Some(5003.0));

    // Mnemonic with spaces around dot (DEPT . FT)
    assert!(well.has_curve("DEPT"));
    assert_eq!(well.curve_items[0].unit(), "FT");

    // Parameter values with colons
    assert_eq!(
        well.get_header_value(SectionType::Parameter, "TIME"),
        Some("14:35:00")
    );
    assert_eq!(
        well.get_header_value(SectionType::Parameter, "SCALE"),
        Some("1:240")
    );

    // Parameter description with colon
    let run_item = well
        .get_header_item(SectionType::Parameter, "RUN")
        .expect("Missing RUN");
    assert_eq!(run_item.value(), "2");
    assert_eq!(run_item.description(), "MAIN PASS : FROM 5000 TO 5003 FT");

    // Missing optional ~O section produces empty list
    assert!(well.other().is_empty());

    // Numerical curve data
    let gr = well.curve_data_by_name("GR").unwrap();
    assert_eq!(gr[0], 62.5);
    assert_eq!(gr[1], 65.1);
    assert_eq!(gr[2], 68.4);
    assert!(gr[3].is_nan()); // -999.25
    assert!(is_quiet_nan(gr[3]));
}

#[test]
fn test_channel_ordering_and_zero_copy_bytes() {
    let well = parse_las_bytes(
        CWLS_12_CANONICAL.as_bytes(),
        Some(ValidationMode::Permissive),
        None,
        OutputFormat::List,
    )
    .unwrap();

    // Channel order must strictly match curve_items
    for (i, name) in well.curve_names().iter().enumerate() {
        assert_eq!(&well.curve_items[i].mnemonic(), name);
        assert_eq!(well.resolve_mnemonic(name), Some(i));

        // Raw bytes slice must directly point to the contiguous f64 memory
        let bytes = well.raw_curve_bytes(i).unwrap();
        assert_eq!(bytes.len(), well.num_rows() * std::mem::size_of::<f64>());

        let col_slice = well.curve_data(i).unwrap();
        assert_eq!(col_slice.as_ptr().cast::<u8>(), bytes.as_ptr());
    }
}
