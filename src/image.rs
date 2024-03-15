use crate::{
    colour::{enum_to_bitmap, get_channel_and_others, symmetric_pair_index, Colour},
    error::ProcessorError,
    DRAW_BORDER, FILL_UNHIGHLIGHTED, WRITE_TO_VIDEO,
};
use itertools::Itertools;
use lazy_static::lazy_static;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{cmp::Ordering, process};

pub static BORDERS: (usize, usize, usize, usize) = (30, 120, 30, 120); // Left    Top    Right    Bottom
pub static CIRC_RADIUS: usize = 18; // 18
pub static CIRC_INNER_RADIUS: usize = 4;
lazy_static! {
    pub static ref ALL_COLOURS: Vec<Colour> = vec![
        Colour::BLUE,
        Colour::RED,
        Colour::YELLOW,
        Colour::RED,
        Colour::BLUE,
    ];
    pub static ref EPSILON_THRESHOLD: Vec<(u8, u8)> = vec![(40, 25), (50, 40), (80, 70)];
    pub static ref SELECTED_BITMAPS: Vec<[bool; 3]> = ALL_COLOURS
        .iter()
        .map(|&colour| enum_to_bitmap(colour))
        .collect();
    pub static ref SELECTED_VALUES: Vec<(Vec<usize>, Vec<usize>, Vec<u8>)> = SELECTED_BITMAPS
        .iter()
        .map(|bitmap| match get_channel_and_others(bitmap) {
            Ok(tup) => tup,
            Err(err) => {
                eprintln!("{err:?}");
                process::exit(0x1000);
            }
        })
        .collect();

    #[derive(Debug)]
    pub static ref SYMMETRY_AROUND: usize = {
        let mut rest = ALL_COLOURS.clone();
        let mut test = vec![];
        ALL_COLOURS.iter().for_each(|&x| {
            if !test.contains(&(true, x)) && !test.contains(&(false, x)) {
                test.push((true, x));
                rest.remove(
                    if let Some((value, _)) = rest.iter().find_position(|&&y| y == x) {
                        value
                    } else {
                        eprintln!("Could not find centre of symmetry");
                        process::exit(0x1000);
                    },
                );

                if rest.contains(&x) {
                    test.pop();
                    test.push((false, x));
                }
            }
        });

        test.iter()
            .filter_map(|&(keep, x)| keep.then_some(x))
            .filter_map(|col| {
                let Some((index, _)) = ALL_COLOURS.iter().find_position(|&&y| y == col) else {
                    return None;
                };

                (index > 0
                    && index < ALL_COLOURS.len() - 1
                    && ALL_COLOURS[index - 1] == ALL_COLOURS[index + 1])
                    .then_some(index)
            })
            .collect::<Vec<_>>().first().map_or_else(|| test.len() - 1, |&index| index)
    };
}

pub fn get_highlighted_pixels(
    image: &mut ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 3]>>,
    image_size: (u32, u32),
) -> Result<Vec<(usize, (usize, usize))>, ProcessorError> {
    ndarray::Zip::indexed(image.rows_mut())
        .into_par_iter()
        .map(|(i, mut pix)| {
            let max_colour = SELECTED_VALUES
                .iter()
                .take(*SYMMETRY_AROUND + 1)
                .enumerate()
                .filter_map(|(colour_index, (selected_channels, other_channels, _))| {
                    if (BORDERS.1..=(image_size.1 as usize - BORDERS.3)).contains(&i.0)
                        && (BORDERS.0..=(image_size.0 as usize - BORDERS.2)).contains(&i.1)
                    {
                        let balanced = (selected_channels
                            .iter()
                            .fold(0, |acc, &colour_index| acc + pix[colour_index] as usize)
                            / selected_channels.len()) as u8;

                        let balanced_others = (other_channels
                            .iter()
                            .fold(0, |acc, &colour_index| acc + pix[colour_index] as usize)
                            / other_channels.len())
                            as u8;

                        (balanced_others.saturating_add(EPSILON_THRESHOLD[colour_index].1)
                            < balanced
                            && balanced > EPSILON_THRESHOLD[colour_index].0)
                            .then_some((colour_index, (i, balanced, balanced_others)))
                    } else {
                        // Output for outside of border
                        if DRAW_BORDER && WRITE_TO_VIDEO {
                            (pix[0], pix[1], pix[2]) = (255, 255, 255);
                        }

                        None
                    }
                })
                .max_by_key(|(_, x)| x.1.saturating_sub(x.2));

            if let Some((colour_index, ((y, x), _, _))) = max_colour {
                // Highlight selected pixels
                if WRITE_TO_VIDEO {
                    let set_colour = SELECTED_VALUES[colour_index].2.clone();
                    (pix[0], pix[1], pix[2]) = (set_colour[0], set_colour[1], set_colour[2]);
                };

                Ok(Some((colour_index, (y, x))))
            } else {
                // Colour for unmatched pixels
                if WRITE_TO_VIDEO && FILL_UNHIGHLIGHTED {
                    (pix[0], pix[1], pix[2]) = (0, 0, 0);
                };

                Ok(None)
            }
        })
        .filter_map(|res| match res {
            Ok(Some(value)) => Some(Ok(value)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        })
        .collect()
}

pub fn get_avg_pos(
    highlighted_pixels: &[(usize, (usize, usize))],
    image_size: (u32, u32),
) -> Vec<(usize, usize)> {
    highlighted_pixels
        .iter()
        .fold(
            vec![(0, (0, 0)); ALL_COLOURS.len()],
            |mut acc, &(colour_index, (y, x))| {
                match colour_index.cmp(&SYMMETRY_AROUND) {
                    Ordering::Less => {
                        // If current key is not for symmetric centre, then split based on which half of the screen the colour is
                        if x < image_size.0 as usize / 2 {
                            acc[colour_index].1 .0 += y;
                            acc[colour_index].1 .1 += x;

                            // Counts
                            acc[colour_index].0 += 1;
                        } else {
                            // Add to symmetric partner (if there is one)
                            if let Ok(new_key) = symmetric_pair_index(colour_index) {
                                acc[new_key].1 .0 += y;
                                acc[new_key].1 .1 += x;

                                // Counts
                                acc[new_key].0 += 1;
                            };
                        }
                    }
                    Ordering::Equal => {
                        acc[colour_index].1 .0 += y;
                        acc[colour_index].1 .1 += x;

                        // Counts
                        acc[colour_index].0 += 1;
                    }
                    Ordering::Greater => {
                        // Do nothing (Should never be here)
                    }
                };

                acc
            },
        )
        .iter()
        .map(|&(n, (y, x))| {
            if n != 0 {
                (y.saturating_div(n), x.saturating_div(n))
            } else {
                (y, x)
            }
        })
        .collect()
}

pub fn get_stddev(
    avg_pos: &[(usize, usize)],
    highlighted_pixels: &[(usize, (usize, usize))],
    image_size: (u32, u32),
) -> Result<Vec<usize>, ProcessorError> {
    Ok(highlighted_pixels
        .iter()
        .try_fold(
            vec![(0, 0); ALL_COLOURS.len()],
            |mut acc, &(key, (y, x))| -> Result<_, ProcessorError> {
                let mut index = key;

                if key < *SYMMETRY_AROUND && x >= image_size.0 as usize / 2 {
                    if let Ok(new_index) = symmetric_pair_index(key) {
                        index = new_index;
                    }
                }

                acc[index].0 += (x.saturating_sub(avg_pos[index].1)).pow(2)
                    + (y.saturating_sub(avg_pos[index].0)).pow(2);
                acc[index].1 += 1;

                Ok(acc)
            },
        )?
        .into_par_iter()
        .map(|(diff, n)| {
            if n != 0 {
                (diff.saturating_div(n) as f32).sqrt() as usize
            } else {
                diff
            }
        })
        .collect())
}

pub fn process(
    input_image: ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 3]>>,
    image_size: (u32, u32),
) -> Result<
    (
        ndarray::ArrayBase<ndarray::OwnedRepr<u8>, ndarray::Dim<[usize; 3]>>,
        Vec<(usize, usize)>,
        Vec<usize>,
    ),
    ProcessorError,
> {
    let mut image = input_image;
    let highlighted_pixels = get_highlighted_pixels(&mut image, image_size)?;
    let avg_pos = get_avg_pos(&highlighted_pixels, image_size);
    let stddev = get_stddev(&avg_pos, &highlighted_pixels, image_size)?;

    // Draw avg_pos for each colour --------------------------------------------------------
    if WRITE_TO_VIDEO {
        ndarray::Zip::indexed(image.rows_mut()).par_for_each(|i, mut pix| {
            avg_pos.iter().enumerate().for_each(|(pos_index, &(y, x))| {
                let colour = SELECTED_VALUES[pos_index].2.clone();

                let circ_rad = ((i.1 as isize - x as isize).saturating_pow(2))
                    .saturating_add((i.0 as isize - y as isize).saturating_pow(2))
                    as usize;

                if circ_rad < CIRC_RADIUS.saturating_pow(2)
                    && circ_rad > (CIRC_RADIUS - CIRC_INNER_RADIUS).saturating_pow(2)
                {
                    (pix[0], pix[1], pix[2]) = (colour[0] / 2, colour[1] / 2, colour[2] / 2);
                }
            });
        });
    }

    Ok((image, avg_pos, stddev))
}
