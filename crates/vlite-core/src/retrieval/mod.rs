use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::document::SearchHit;
use crate::metadata::Metadata;

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub top_k: usize,
    pub where_filter: Option<Metadata>,
}

pub trait EmbedderAdapter {
    fn dimensions(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
}

#[derive(Debug, Clone)]
pub struct HashedEmbedder {
    dimensions: usize,
}

impl HashedEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl EmbedderAdapter for HashedEmbedder {
    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; self.dimensions];

        for (position, token) in text.split_whitespace().enumerate() {
            if token.is_empty() {
                continue;
            }

            let hash = xxh3_64_with_seed(token.as_bytes(), position as u64);
            let index = (hash % self.dimensions as u64) as usize;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }

        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }

        vector
    }
}

pub trait Retriever {
    fn search(&self, request: SearchRequest) -> anyhow::Result<Vec<SearchHit>>;
}
