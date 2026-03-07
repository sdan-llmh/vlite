//! Embedding engine — ONNX Runtime text (and optional vision) inference.
//!
//! Two structs, no traits:
//! - `Embedder`: text → vector (sentence-transformer or CLIP text encoder)
//! - `VisionEmbedder`: image bytes → vector (CLIP vision encoder)

use crate::error::{Result, VLiteError};
use ort::session::Session;
use ort::value::Tensor;
use std::cell::UnsafeCell;
use std::path::Path;

// ---------------------------------------------------------------------------
// Text embedder
// ---------------------------------------------------------------------------

/// Wrapper around Session that allows &self inference.
/// ONNX Runtime sessions are thread-safe for inference. ort's Rust API
/// requires &mut self for `run()`, but the underlying C API is safe for
/// concurrent reads. We use UnsafeCell to opt out of the borrow checker here.
struct InferSession(UnsafeCell<Session>);

// SAFETY: ONNX Runtime C API sessions are thread-safe for inference.
unsafe impl Send for InferSession {}
unsafe impl Sync for InferSession {}

impl InferSession {
    fn new(session: Session) -> Self {
        Self(UnsafeCell::new(session))
    }
    fn get_mut(&self) -> &mut Session {
        unsafe { &mut *self.0.get() }
    }
}

pub struct Embedder {
    session: InferSession,
    tokenizer: tokenizers::Tokenizer,
    pub dim: usize,
    pub max_tokens: usize,
    model_name: String,
}

impl Embedder {
    /// Default text model: all-MiniLM-L6-v2 (384-dim, ~80MB ONNX).
    /// Auto-downloads from HuggingFace Hub on first use.
    pub fn new() -> Result<Self> {
        Self::from_hf("sentence-transformers/all-MiniLM-L6-v2", 384, 256)
    }

    /// CLIP text encoder for multimodal mode.
    pub fn clip_text() -> Result<Self> {
        Self::from_hf("openai/clip-vit-base-patch32", 512, 77)
    }

    fn from_hf(repo_id: &str, dim: usize, max_tokens: usize) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| VLiteError::Model(e.to_string()))?;
        let repo = api.model(repo_id.to_string());

        let model_path = repo
            .get("onnx/model.onnx")
            .or_else(|_| repo.get("model.onnx"))
            .map_err(|e| VLiteError::Model(format!("cannot download ONNX from {repo_id}: {e}")))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| VLiteError::Model(format!("cannot download tokenizer from {repo_id}: {e}")))?;

        Self::from_onnx(&model_path, &tokenizer_path, dim, max_tokens, repo_id)
    }

    /// Load from local ONNX + tokenizer files.
    pub fn from_onnx(
        model_path: &Path,
        tokenizer_path: &Path,
        dim: usize,
        max_tokens: usize,
        name: &str,
    ) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| VLiteError::Model(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| VLiteError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| VLiteError::Model(e.to_string()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| VLiteError::Model(e.to_string()))?;

        Ok(Self { session: InferSession::new(session), tokenizer, dim, max_tokens, model_name: name.to_string() })
    }

    /// Embed a single text → f32 vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().unwrap())
    }

    /// Batch embed for throughput.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| VLiteError::Embed(e.to_string()))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        // Build padded input tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                input_ids[i * max_len + j] = id as i64;
            }
            for (j, &m) in enc.get_attention_mask().iter().enumerate() {
                attention_mask[i * max_len + j] = m as i64;
            }
            for (j, &t) in enc.get_type_ids().iter().enumerate() {
                token_type_ids[i * max_len + j] = t as i64;
            }
        }

        let ids_t = Tensor::from_array(([batch_size, max_len], input_ids))
            .map_err(|e| VLiteError::Embed(e.to_string()))?;
        let mask_t = Tensor::from_array(([batch_size, max_len], attention_mask.clone()))
            .map_err(|e| VLiteError::Embed(e.to_string()))?;
        let ttids_t = Tensor::from_array(([batch_size, max_len], token_type_ids))
            .map_err(|e| VLiteError::Embed(e.to_string()))?;

        // Build named inputs as Vec<(&str, Value)> — uses From<Vec<(K,V)>> impl
        let inputs = vec![
            ("input_ids", &ids_t),
            ("attention_mask", &mask_t),
            ("token_type_ids", &ttids_t),
        ];

        let outputs = self.session.get_mut()
            .run(inputs)
            .map_err(|e| VLiteError::Embed(e.to_string()))?;

        // Extract hidden states
        let (shape, hidden_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| VLiteError::Embed(e.to_string()))?;

        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();

        if dims.len() == 2 {
            // Already pooled [batch, dim]
            let dim = dims[1];
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let start = i * dim;
                let mut vec: Vec<f32> = hidden_data[start..start + dim].to_vec();
                l2_normalize(&mut vec);
                results.push(vec);
            }
            return Ok(results);
        }

        // Shape [batch, seq_len, dim] — need mean pooling
        let seq_len = dims[1];
        let dim = dims[2];
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut pooled = vec![0.0f32; dim];
            let mut count = 0.0f32;
            for t in 0..seq_len {
                if t < max_len && attention_mask[i * max_len + t] == 1 {
                    let offset = i * seq_len * dim + t * dim;
                    for d in 0..dim {
                        pooled[d] += hidden_data[offset + d];
                    }
                    count += 1.0;
                }
            }
            if count > 0.0 {
                for d in 0..dim { pooled[d] /= count; }
            }
            l2_normalize(&mut pooled);
            results.push(pooled);
        }

        Ok(results)
    }

    /// Rough token count (used to decide chunking).
    pub fn token_count(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().len())
            .unwrap_or(text.len() / 4)
    }

    pub fn model_name(&self) -> &str { &self.model_name }
}

// ---------------------------------------------------------------------------
// Vision embedder (CLIP)
// ---------------------------------------------------------------------------

pub struct VisionEmbedder {
    session: InferSession,
    pub dim: usize,
}

impl VisionEmbedder {
    /// CLIP ViT-B/32 vision encoder.
    pub fn clip_vision() -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| VLiteError::Model(e.to_string()))?;
        let repo = api.model("openai/clip-vit-base-patch32".to_string());

        let model_path = repo
            .get("onnx/vision_model.onnx")
            .or_else(|_| repo.get("visual_model.onnx"))
            .map_err(|e| VLiteError::Model(format!("cannot download CLIP vision: {e}")))?;

        let session = Session::builder()
            .map_err(|e| VLiteError::Model(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| VLiteError::Model(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| VLiteError::Model(e.to_string()))?;

        Ok(Self { session: InferSession::new(session), dim: 512 })
    }

    /// Embed an image (JPEG/PNG/WebP bytes) → f32 vector.
    pub fn embed_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>> {
        use image::GenericImageView;

        let img = image::load_from_memory(image_bytes)
            .map_err(|e| VLiteError::Embed(format!("image decode: {e}")))?;
        let resized = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);

        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let std_dev = [0.26862954f32, 0.26130258, 0.27577711];
        let mut pixels = vec![0.0f32; 3 * 224 * 224];
        for y in 0..224u32 {
            for x in 0..224u32 {
                let p = resized.get_pixel(x, y);
                for c in 0..3 {
                    pixels[c * 224 * 224 + y as usize * 224 + x as usize] =
                        (p[c] as f32 / 255.0 - mean[c]) / std_dev[c];
                }
            }
        }

        let input_t = Tensor::from_array(([1usize, 3, 224, 224], pixels))
            .map_err(|e| VLiteError::Embed(e.to_string()))?;

        let outputs = self.session.get_mut()
            .run(vec![("pixel_values", &input_t)])
            .map_err(|e| VLiteError::Embed(e.to_string()))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| VLiteError::Embed(e.to_string()))?;

        let mut vec: Vec<f32> = data.iter().copied().take(self.dim).collect();
        l2_normalize(&mut vec);
        Ok(vec)
    }
}

// ---------------------------------------------------------------------------
// Cross-encoder reranker (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "rerank")]
pub struct Reranker {
    session: InferSession,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "rerank")]
impl Reranker {
    pub fn new() -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| VLiteError::Model(e.to_string()))?;
        let repo = api.model("mixedbread-ai/mxbai-rerank-xsmall-v1".to_string());

        let model_path = repo
            .get("onnx/model_quantized.onnx")
            .or_else(|_| repo.get("onnx/model.onnx"))
            .map_err(|e| VLiteError::Model(format!("cannot download reranker: {e}")))?;
        let tok_path = repo
            .get("tokenizer.json")
            .map_err(|e| VLiteError::Model(format!("cannot download reranker tok: {e}")))?;

        let session = Session::builder()
            .map_err(|e| VLiteError::Model(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| VLiteError::Model(e.to_string()))?;
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| VLiteError::Model(e.to_string()))?;

        Ok(Self { session: InferSession::new(session), tokenizer })
    }

    pub fn rerank(
        &self, query: &str, candidates: &[(usize, f32)], texts: &[String],
    ) -> Result<Vec<(usize, f32)>> {
        let mut scored = Vec::with_capacity(candidates.len());
        for &(idx, _) in candidates {
            let pair = format!("{query} [SEP] {}", &texts[idx]);
            let enc = self.tokenizer.encode(pair.as_str(), true)
                .map_err(|e| VLiteError::Embed(e.to_string()))?;
            let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
            let mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();
            let len = ids.len();

            let ids_t = Tensor::from_array(([1usize, len], ids))
                .map_err(|e| VLiteError::Embed(e.to_string()))?;
            let mask_t = Tensor::from_array(([1usize, len], mask))
                .map_err(|e| VLiteError::Embed(e.to_string()))?;

            let outputs = self.session.get_mut()
                .run(vec![("input_ids", &ids_t), ("attention_mask", &mask_t)])
                .map_err(|e| VLiteError::Embed(e.to_string()))?;

            let (_shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| VLiteError::Embed(e.to_string()))?;

            scored.push((idx, data.first().copied().unwrap_or(0.0)));
        }
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

// ---------------------------------------------------------------------------

fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for v in vec.iter_mut() { *v /= norm; }
    }
}
