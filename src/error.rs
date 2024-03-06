use std::io;

use thiserror::Error;

use crate::colour::Colour;

#[derive(Error, Debug)]
pub enum ProcessorError {
    #[error("Error when selecting channels:\n\tSelected Indices: {selected_channels_indices:?}\n\tInput: {selected_channels_bit_array:?} [Size: {}]", selected_channels_bit_array.len())]
    ChannelSelectErr {
        selected_channels_indices: Vec<usize>,
        selected_channels_bit_array: Vec<bool>,
    },

    #[error("Decoder Error: {e}")]
    DecoderErr { e: String },

    #[error("Could not perform IO")]
    IoErr(#[from] io::Error),

    #[error("Could not create path: [\"{path_name}\"]")]
    PathErr { path_name: String },

    #[error("Could not launch plotting program")]
    PlotLaunchErr,

    #[error("Could not save plot to file: {err}")]
    PlotSaveErr { err: String },

    #[error("Could not display message to terminal")]
    TermDisplayErr(#[from] term::Error),

    #[error("Trying to match colour with too many channels: [{channel_amt}]")]
    TooManyChannelsErr { channel_amt: usize },

    #[error("Could not launch video player")]
    VideoPlayerLaunchErr(#[from] video_rs::Error),

    #[error("Video_RS Initialisation Error")]
    VideoRsErr { err: String },

    #[error("Could not convert index to colour [{index}]")]
    EnumToIndexErr { index: usize },

    #[error("Could not convert colour to index [{colour:?}]")]
    IndexToEnumErr { colour: Colour },

    #[error("Could not convert data to JSON")]
    JsonConvertErr(#[from] serde_json::Error),

    #[error("Could not find symmetric pair: {index}")]
    SymmetricPairErr { index: usize },
}
