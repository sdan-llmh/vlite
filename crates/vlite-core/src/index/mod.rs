use crate::document::Segment;
use crate::metadata::{matches_filter, Metadata};

pub trait IndexBackend {
    fn search(
        &self,
        query: &[f32],
        segments: &[Segment],
        top_k: usize,
        where_filter: Option<&Metadata>,
    ) -> Vec<(usize, f32)>;
}

#[derive(Debug, Default, Clone)]
pub struct ExactIndex;

impl ExactIndex {
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() || a.len() != b.len() {
            return 0.0;
        }

        let mut dot = 0.0_f32;
        let mut a_norm = 0.0_f32;
        let mut b_norm = 0.0_f32;

        for (lhs, rhs) in a.iter().zip(b.iter()) {
            dot += lhs * rhs;
            a_norm += lhs * lhs;
            b_norm += rhs * rhs;
        }

        if a_norm == 0.0 || b_norm == 0.0 {
            return 0.0;
        }

        dot / (a_norm.sqrt() * b_norm.sqrt())
    }
}

impl IndexBackend for ExactIndex {
    fn search(
        &self,
        query: &[f32],
        segments: &[Segment],
        top_k: usize,
        where_filter: Option<&Metadata>,
    ) -> Vec<(usize, f32)> {
        let mut scored = segments
            .iter()
            .enumerate()
            .filter(|(_, segment)| segment.searchable)
            .filter(|(_, segment)| matches_filter(&segment.metadata, where_filter))
            .map(|(idx, segment)| (idx, Self::cosine_similarity(query, &segment.embedding)))
            .collect::<Vec<_>>();

        scored.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
        scored.truncate(top_k);
        scored
    }
}
