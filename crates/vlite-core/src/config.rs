use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChunkingStrategy {
    Auto,
    Structural,
    Late,
    Vision,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetrievalStrategy {
    Dense,
    Hybrid,
    LateInteraction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VLiteConfig {
    pub chunking_strategy: ChunkingStrategy,
    pub retrieval_strategy: RetrievalStrategy,
    pub embedding_dimensions: usize,
    pub store_raw: bool,
}

impl Default for VLiteConfig {
    fn default() -> Self {
        Self {
            chunking_strategy: ChunkingStrategy::Auto,
            retrieval_strategy: RetrievalStrategy::Hybrid,
            embedding_dimensions: 128,
            store_raw: true,
        }
    }
}
