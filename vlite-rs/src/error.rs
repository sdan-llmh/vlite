//! Error types for vlite.

#[derive(Debug, thiserror::Error)]
pub enum VLiteError {
    #[error("model error: {0}")]
    Model(String),

    #[error("embedding error: {0}")]
    Embed(String),

    #[error("io error: {0}")]
    Io(String),

    #[error("no vision model loaded — create with VLite::with_clip()")]
    NoVisionModel,

    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),

    #[error("empty database — nothing to search")]
    EmptyDatabase,
}

pub type Result<T> = std::result::Result<T, VLiteError>;
