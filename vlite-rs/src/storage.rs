//! .vlite binary file format — save/open.
//!
//! Format (v2):
//!   Header (24 bytes): magic + version + dim + count + flags + avg_doc_len
//!   Vectors:   [f32; dim × count]
//!   Texts:     (u32 len + UTF-8) × count
//!   Parents:   (u32 len + UTF-8) × count
//!   Metadata:  (u32 len + JSON)  × count
//!   DocLens:   [u32; count]
//!   DF Map:    u32 entry_count + (u32 len + UTF-8 + u32 freq) × entries

use crate::error::{Result, VLiteError};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"VLIT";
const VERSION: u32 = 2;

/// Flags
pub const FLAG_HAS_VISION: u32 = 1 << 0;

/// All the state needed to reconstruct a VLite instance.
pub struct SavedState {
    pub dim: usize,
    pub flags: u32,
    pub avg_doc_len: f32,
    pub vectors: Vec<f32>,
    pub texts: Vec<String>,
    pub parents: Vec<String>,
    pub metadata: Vec<HashMap<String, serde_json::Value>>,
    pub doc_lens: Vec<usize>,
    pub df: HashMap<String, usize>,
}

pub fn save(state: &SavedState, path: &Path) -> Result<()> {
    let count = state.texts.len();
    let mut f = std::fs::File::create(path).map_err(|e| VLiteError::Io(e.to_string()))?;

    // Header (24 bytes)
    f.write_all(MAGIC).map_err(io_err)?;
    f.write_all(&VERSION.to_le_bytes()).map_err(io_err)?;
    f.write_all(&(state.dim as u32).to_le_bytes()).map_err(io_err)?;
    f.write_all(&(count as u32).to_le_bytes()).map_err(io_err)?;
    f.write_all(&state.flags.to_le_bytes()).map_err(io_err)?;
    f.write_all(&state.avg_doc_len.to_le_bytes()).map_err(io_err)?;

    // Vectors — raw f32 bytes
    if !state.vectors.is_empty() {
        let vec_bytes: &[u8] = bytemuck::cast_slice(&state.vectors);
        f.write_all(vec_bytes).map_err(io_err)?;
    }

    // Texts
    for text in &state.texts {
        write_length_prefixed_str(&mut f, text)?;
    }
    // Parents
    for parent in &state.parents {
        write_length_prefixed_str(&mut f, parent)?;
    }
    // Metadata
    for meta in &state.metadata {
        let json = serde_json::to_vec(meta).map_err(|e| VLiteError::Io(e.to_string()))?;
        f.write_all(&(json.len() as u32).to_le_bytes()).map_err(io_err)?;
        f.write_all(&json).map_err(io_err)?;
    }
    // DocLens
    for &dl in &state.doc_lens {
        f.write_all(&(dl as u32).to_le_bytes()).map_err(io_err)?;
    }
    // DF map
    f.write_all(&(state.df.len() as u32).to_le_bytes()).map_err(io_err)?;
    for (term, &freq) in &state.df {
        write_length_prefixed_str(&mut f, term)?;
        f.write_all(&(freq as u32).to_le_bytes()).map_err(io_err)?;
    }

    f.flush().map_err(io_err)?;
    Ok(())
}

pub fn load(path: &Path) -> Result<SavedState> {
    let mut f = std::fs::File::open(path).map_err(|e| VLiteError::Io(e.to_string()))?;

    // Header
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(io_err)?;
    if &magic != MAGIC {
        return Err(VLiteError::Io("invalid .vlite file (bad magic)".into()));
    }

    let version = read_u32(&mut f)?;
    if version > VERSION {
        return Err(VLiteError::Io(format!(
            "unsupported .vlite version {version} (max {VERSION})"
        )));
    }

    let dim = read_u32(&mut f)? as usize;
    let count = read_u32(&mut f)? as usize;
    let flags = read_u32(&mut f)?;
    let avg_doc_len = read_f32(&mut f)?;

    // Vectors
    let num_floats = dim * count;
    let vectors: Vec<f32> = if num_floats > 0 {
        let mut vec_bytes = vec![0u8; num_floats * 4];
        f.read_exact(&mut vec_bytes).map_err(io_err)?;
        bytemuck::cast_slice(&vec_bytes).to_vec()
    } else {
        vec![]
    };

    // Texts
    let mut texts = Vec::with_capacity(count);
    for _ in 0..count {
        texts.push(read_length_prefixed_str(&mut f)?);
    }
    // Parents
    let mut parents = Vec::with_capacity(count);
    for _ in 0..count {
        parents.push(read_length_prefixed_str(&mut f)?);
    }
    // Metadata
    let mut metadata = Vec::with_capacity(count);
    for _ in 0..count {
        let len = read_u32(&mut f)? as usize;
        if len == 0 {
            metadata.push(HashMap::new());
        } else {
            let mut buf = vec![0u8; len];
            f.read_exact(&mut buf).map_err(io_err)?;
            let m: HashMap<String, serde_json::Value> =
                serde_json::from_slice(&buf).map_err(|e| VLiteError::Io(e.to_string()))?;
            metadata.push(m);
        }
    }
    // DocLens
    let mut doc_lens = Vec::with_capacity(count);
    for _ in 0..count {
        doc_lens.push(read_u32(&mut f)? as usize);
    }
    // DF map
    let df_count = read_u32(&mut f)? as usize;
    let mut df = HashMap::with_capacity(df_count);
    for _ in 0..df_count {
        let term = read_length_prefixed_str(&mut f)?;
        let freq = read_u32(&mut f)? as usize;
        df.insert(term, freq);
    }

    Ok(SavedState {
        dim,
        flags,
        avg_doc_len,
        vectors,
        texts,
        parents,
        metadata,
        doc_lens,
        df,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn io_err(e: std::io::Error) -> VLiteError {
    VLiteError::Io(e.to_string())
}

fn write_length_prefixed_str(f: &mut std::fs::File, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    f.write_all(&(bytes.len() as u32).to_le_bytes()).map_err(io_err)?;
    f.write_all(bytes).map_err(io_err)?;
    Ok(())
}

fn read_length_prefixed_str(f: &mut std::fs::File) -> Result<String> {
    let len = read_u32(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf).map_err(io_err)?;
    String::from_utf8(buf).map_err(|e| VLiteError::Io(e.to_string()))
}

fn read_u32(f: &mut std::fs::File) -> Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).map_err(io_err)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f32(f: &mut std::fs::File) -> Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).map_err(io_err)?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vlite");

        let state = SavedState {
            dim: 4,
            flags: 0,
            avg_doc_len: 5.0,
            vectors: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            texts: vec!["hello".into(), "world".into()],
            parents: vec!["hello parent".into(), "world parent".into()],
            metadata: vec![HashMap::new(), HashMap::new()],
            doc_lens: vec![1, 1],
            df: {
                let mut m = HashMap::new();
                m.insert("hello".into(), 1);
                m.insert("world".into(), 1);
                m
            },
        };

        save(&state, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.dim, 4);
        assert_eq!(loaded.vectors, state.vectors);
        assert_eq!(loaded.texts, state.texts);
        assert_eq!(loaded.parents, state.parents);
        assert_eq!(loaded.doc_lens, state.doc_lens);
        assert_eq!(loaded.df.len(), 2);
    }
}
