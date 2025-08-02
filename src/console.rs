use std::{env, fs, path::PathBuf, time::Instant};
use term::color::{self, Color};
use video_rs::{encode::Settings, Decoder, Encoder, Location};

use crate::error::ProcessorError;

pub const HIGHLIGHT_COLOUR: Color = color::BRIGHT_GREEN;

pub fn process_args(
    term: &mut Box<term::StdoutTerminal>,
) -> Result<(Decoder, Encoder, (u32, u32), (String, String, String)), ProcessorError> {
    let source_name_default: String = "res/video".to_owned();
    let ext_default: String = ".mp4".to_owned();
    let destination_default: String = "data/output".to_string();

    let source_folder: String = "res".to_string();

    let destination_folder: String = "data".to_string();

    let args: Vec<String> = env::args().collect();
    let (source, (dest_no_name, full_destination), source_name, source_ext) = if args.len() == 2 {
        let source = args[1].clone();
        let source_path = PathBuf::from(source.clone());

        let mut filename_with_ext = String::new();

        let dest_path = source_path.as_path().iter().enumerate().try_fold(
            String::new(),
            |mut acc, (i, path_element)| -> Result<_, ProcessorError> {
                let element_str = path_element.to_str().map_or_else(
                    || {
                        Err(ProcessorError::PathErr {
                            path_name: "Path element not found".to_string(),
                        })
                    },
                    |el| Ok(el.to_string()),
                )?;

                if i < source_path.as_path().iter().count() - 1 {
                    if element_str == source_folder {
                        acc.push_str(destination_folder.as_str());
                    } else {
                        acc.push_str(element_str.as_str());
                    };

                    if i < source_path.as_path().iter().count() - 2 {
                        acc.push('/');
                    }
                } else {
                    filename_with_ext = element_str;
                };

                Ok(acc)
            },
        )?;

        let (filename, source_ext) = filename_with_ext.split_once('.').map_or_else(
            || {
                Err(ProcessorError::PathErr {
                    path_name: filename_with_ext.clone(),
                })
            },
            Ok,
        )?;
        let destination = format!("{dest_path}/{filename}-output.{source_ext}");

        fs::create_dir_all(dest_path.clone())?;

        // let (source_name, source_ext) = (
        //     match source_path.as_path().file_stem() {
        //         Some(path_osstr) => match path_osstr.to_str() {
        //             Some(path) => path.to_string(),
        //             None => return Err(ProcessorError::PathErr { path_name: source }),
        //         },
        //         None => return Err(ProcessorError::PathErr { path_name: source }),
        //     },
        //     {
        //         match source_path.as_path().extension() {
        //             Some(ext) => match ext.to_str() {
        //                 Some(ext_str) => ext_str.to_string(),
        //                 None => return Err(ProcessorError::PathErr { path_name: source }),
        //             },
        //             None => return Err(ProcessorError::PathErr { path_name: source }),
        //         }
        //     },
        // );

        // let destination = format!("{destination_folder}/{source_name}-output.{source_ext}");

        (source, (dest_path, destination), filename.to_string(), source_ext.to_string())
    } else {
        println!("Using default source and destination video filepaths.");
        (
            format!("{source_name_default}.{ext_default}"),
            (destination_folder, destination_default),
            source_name_default,
            ext_default,
        )
    };

    term.reset()?;
    write!(term, "Source: ")?;
    term.fg(HIGHLIGHT_COLOUR)?;
    write!(term, "{source}")?;
    term.reset()?;
    write!(term, "    Destination: ")?;
    term.fg(HIGHLIGHT_COLOUR)?;
    write!(term, "{full_destination}")?;
    term.reset()?;

    match video_rs::init() {
        Ok(it) => it,
        Err(err) => return Err(ProcessorError::VideoRsErr { err: err.to_string() }),
    };

    let decoder = Decoder::new(&Into::<Location>::into(PathBuf::from(source)))?;
    let size = decoder.size();

    writeln!(term, "    {size:?}\n\n\n",)?;

    let encoder = Encoder::new(
        &Into::<Location>::into(PathBuf::from(full_destination)),
        Settings::preset_h264_yuv420p(size.0 as usize, size.1 as usize, false),
    )?;

    Ok((decoder, encoder, size, (source_name, source_ext, dest_no_name)))
}

pub fn print_timing_info(
    term: &mut Box<term::StdoutTerminal>,
    start_time: Instant,
    before_time: Instant,
    video_position: f32,
    iteration_index: usize,
) -> Result<(), ProcessorError> {
    let space_per_section = 15;

    term.cursor_up()?;
    term.cursor_up()?;
    term.reset()?;
    term.delete_line()?;
    writeln!(
        term,
        "{:^space_per_section$}{:^space_per_section$}{:^space_per_section$}{:^space_per_section$}{:^space_per_section$}",
        "Frame", "Video Time", "Actual Time", "Took", "Average"
    )?;

    let time_now = Instant::now();

    term.delete_line()?;
    term.fg(HIGHLIGHT_COLOUR)?;
    writeln!(
        term,
        "{:^space_per_section$}{:^space_per_section$}{:^space_per_section$}{:^space_per_section$}{:^space_per_section$}",
        iteration_index + 1,
        format!("{:.2}s", video_position),
        format!("{:.2}s", (time_now - start_time).as_secs_f32()),
        format!("{:.2}ms", (time_now - before_time).as_micros() as f32 / 1000f32),
        format!(
            "{:.2}ms",
            ((time_now - start_time).as_micros() as f32 / 1000f32) / (iteration_index as f32 + 1f32)
        )
    )?;
    term.reset()?;

    Ok(())
}

pub fn print_final_timing_info(
    term: &mut Box<term::StdoutTerminal>,
    start_time: Instant,
    final_index: usize,
) -> Result<(), ProcessorError> {
    let current_now = Instant::now();
    let time_elapsed = (current_now - start_time).as_secs_f32();
    let time_elapsed_ms = (current_now - start_time).as_micros() as f32 / 1000f32;

    term.reset()?;
    write!(term, "\nTook ")?;
    term.fg(HIGHLIGHT_COLOUR)?;
    write!(term, "{time_elapsed:.3}s")?;
    term.reset()?;
    write!(term, " to process ")?;
    term.fg(HIGHLIGHT_COLOUR)?;
    write!(term, "{}", final_index + 1)?;
    term.reset()?;
    write!(term, " frames [")?;
    term.fg(HIGHLIGHT_COLOUR)?;
    write!(term, "{:.2}ms", time_elapsed_ms / (final_index as f32 + 1f32))?;
    term.reset()?;
    writeln!(term, " per frame]")?;

    Ok(())
}
