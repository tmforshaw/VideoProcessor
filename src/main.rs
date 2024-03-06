#![warn(
    // clippy::all,
    clippy::restriction,
    clippy::pedantic,
    clippy::nursery,
    // clippy::cargo
)]
// #![warn(
//     clippy::unwrap_used,
//     clippy::clone_on_copy,
//     clippy::unneeded_field_pattern
// )]
#![allow(
    clippy::implicit_return,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::type_complexity,
    clippy::module_name_repetitions,
    clippy::single_call_fn,
    clippy::unseparated_literal_suffix,
    clippy::exhaustive_enums,
    clippy::question_mark_used,
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::str_to_string,
    clippy::print_stdout,
    clippy::use_debug,
    clippy::min_ident_chars,
    clippy::as_conversions,
    clippy::exit,
    clippy::float_arithmetic,
    clippy::arithmetic_side_effects,
    clippy::print_stderr,
    clippy::integer_division,
    clippy::shadow_unrelated,
    clippy::pattern_type_mismatch,
    clippy::std_instead_of_alloc,
    clippy::std_instead_of_core,
    clippy::default_numeric_fallback,
    clippy::blanket_clippy_restriction_lints,
    clippy::too_many_lines
)]

use std::{process::Command, time::Instant};

use term::Error;
use video_rs::Time;

pub mod colour;
mod console;
pub mod error;
mod image;
mod json;

use console::{print_final_timing_info, print_timing_info, process_args};
use error::ProcessorError;
use image::{process, ALL_COLOURS};
use json::positions_to_json;

pub static SHOW_PLOT: bool = true;
pub static WRITE_TO_VIDEO: bool = true;
pub static DRAW_BORDER: bool = false;
pub static FILL_UNHIGHLIGHTED: bool = false;
pub static DRAW_CENTRAL_REGION: bool = false;

pub static FRAMES_TO_TAKE: usize = 600;
pub static FRAMES_TO_SKIP: usize = 0;

/// .
///
/// # Panics
///
/// Panics if .
///
/// # Errors
///
/// This function will return an error if any errors have been returned within the program.
fn main() -> Result<(), ProcessorError> {
    let Some(mut term) = term::stdout() else {
        return Err(ProcessorError::TermDisplayErr(Error::NotSupported));
    };

    let duration = Time::from_nth_of_a_second(24);
    let mut position = Time::zero();

    let (mut decoder, mut encoder, size, (input_name, input_ext, output_folder)) =
        process_args(&mut term)?;

    let mut final_index = 0;

    let start_time = Instant::now();
    let mut before_time = Instant::now();

    let (avg_positions, stddevs) = decoder
        .decode_iter()
        // .skip(FRAMES_TO_SKIP)
        // .take(FRAMES_TO_TAKE)
        .take_while(Result::is_ok)
        .map(Result::unwrap)
        .enumerate()
        .map(|(i, (_, frame))| -> Result<_, ProcessorError> {
            position = position.aligned_with(&duration).add();

            let (frame_new, avg_positions, stddev) = process(frame, size)?;

            if WRITE_TO_VIDEO {
                encoder.encode(&frame_new, &position)?;
            }

            print_timing_info(&mut term, start_time, before_time, position.as_secs(), i)?;

            final_index = i;
            before_time = Instant::now();

            Ok((avg_positions, stddev))
        })
        .try_fold(
            (
                vec![Vec::new(); ALL_COLOURS.len()],
                vec![Vec::new(); ALL_COLOURS.len()],
            ),
            |mut acc, res| -> Result<_, ProcessorError> {
                let (all_positions, stddev) = res?;

                all_positions.iter().enumerate().for_each(|(i, &pos)| {
                    acc.0[i].push(pos);
                    acc.1[i].push(stddev[i]);
                });

                Ok(acc)
            },
        )?;

    print_final_timing_info(&mut term, start_time, final_index)?;

    let json_filename_and_ext = (
        format!("{output_folder}/{input_name}"),
        String::from("-data.json"),
    );

    positions_to_json(&avg_positions, &stddevs, &json_filename_and_ext)?;

    if WRITE_TO_VIDEO {
        Command::new("parole")
            .arg(format!("{output_folder}/{input_name}-output.{input_ext}"))
            .spawn()?;
    }

    if SHOW_PLOT {
        Command::new("src/plot.py")
            .arg(json_filename_and_ext.0)
            .arg(json_filename_and_ext.1)
            .spawn()?;
    }

    Ok(())
}
