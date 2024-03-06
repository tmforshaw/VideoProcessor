use crate::{
    error::ProcessorError,
    image::{ALL_COLOURS, SYMMETRY_AROUND},
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Colour {
    // GREEN,
    BLUE,
    RED,
    YELLOW,
}

impl TryFrom<usize> for Colour {
    type Error = ProcessorError;

    fn try_from(index: usize) -> Result<Self, Self::Error> {
        if index < ALL_COLOURS.len() {
            Ok(ALL_COLOURS[index])
        } else {
            Err(ProcessorError::EnumToIndexErr { index })
        }
    }
}

impl TryInto<usize> for Colour {
    type Error = ProcessorError;
    fn try_into(self) -> Result<usize, Self::Error> {
        ALL_COLOURS
            .iter()
            .enumerate()
            .find_map(|(i, &colour)| (colour == self).then_some(i))
            .map_or_else(|| Err(ProcessorError::IndexToEnumErr { colour: self }), Ok)
    }
}

/// .
///
/// # Errors
///
/// This function will return an error if a symmetric pair doesn't exist.
pub fn symmetric_pair_index(index: usize) -> Result<usize, ProcessorError> {
    match (2 * *SYMMETRY_AROUND).checked_sub(index) {
        Some(i) if i < ALL_COLOURS.len() => Ok(i),
        _ => Err(ProcessorError::SymmetricPairErr { index }),
    }
}

/// .
///
/// # Errors
///
/// This function will return an error if a symmetric pair doesn't exist.
pub fn to_symmetric_pair(index: usize) -> Result<Colour, ProcessorError> {
    Ok(ALL_COLOURS[symmetric_pair_index(index)?])
}

#[must_use]
pub const fn enum_to_bitmap(col: Colour) -> [bool; 3] {
    match col {
        Colour::RED => [true, false, false],
        Colour::YELLOW => [true, true, false],
        Colour::BLUE => [false, true, true],
    }
}

/// .
///
/// # Errors
///
/// This function will return an error if the colour has too many channels
pub fn get_channel_and_others(
    selected_channels_bit_array: &[bool],
) -> Result<(Vec<usize>, Vec<usize>, Vec<u8>), ProcessorError> {
    let selected_channels_indices = selected_channels_bit_array
        .iter()
        .enumerate()
        .filter(|&(_, &value)| value)
        .map(|(index, _)| index)
        .collect::<Vec<usize>>();

    let set_colour = selected_channels_bit_array
        .iter()
        .map(|&boolean| if boolean { 255 } else { 0 })
        .collect();

    let mut others = (0..=2).collect::<Vec<usize>>();

    let selected = match selected_channels_indices.len() {
        1 => {
            others.remove(selected_channels_indices[0]);

            vec![selected_channels_indices[0]]
        }
        2 => {
            others.remove(selected_channels_indices[1]);
            others.remove(selected_channels_indices[0]);

            selected_channels_indices
        }
        _ => {
            return Err(ProcessorError::ChannelSelectErr {
                selected_channels_indices,
                selected_channels_bit_array: selected_channels_bit_array.to_vec(),
            });
        }
    };

    Ok((selected, others, set_colour))
}
