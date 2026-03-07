use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::config::VLiteConfig;
use crate::document::{Document, Segment};

const STORE_FILE: &str = "collection.json";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CollectionSnapshot {
    pub config: Option<VLiteConfig>,
    pub documents: Vec<Document>,
    pub segments: Vec<Segment>,
}

pub trait Store {
    fn load(&self) -> anyhow::Result<CollectionSnapshot>;
    fn save(&self, snapshot: &CollectionSnapshot) -> anyhow::Result<()>;
    fn root(&self) -> &Path;
}

#[derive(Debug, Clone)]
pub struct JsonStore {
    root: PathBuf,
    file_path: PathBuf,
}

impl JsonStore {
    pub fn open(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let root = path.into();
        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create store directory at {}", root.display()))?;
        let file_path = root.join(STORE_FILE);
        Ok(Self { root, file_path })
    }
}

impl Store for JsonStore {
    fn load(&self) -> anyhow::Result<CollectionSnapshot> {
        if !self.file_path.exists() {
            return Ok(CollectionSnapshot::default());
        }

        let raw = fs::read_to_string(&self.file_path).with_context(|| {
            format!(
                "failed to read collection snapshot from {}",
                self.file_path.display()
            )
        })?;
        let snapshot = serde_json::from_str(&raw).with_context(|| {
            format!(
                "failed to deserialize collection snapshot from {}",
                self.file_path.display()
            )
        })?;
        Ok(snapshot)
    }

    fn save(&self, snapshot: &CollectionSnapshot) -> anyhow::Result<()> {
        let raw = serde_json::to_string_pretty(snapshot)
            .context("failed to serialize collection snapshot")?;
        fs::write(&self.file_path, raw).with_context(|| {
            format!(
                "failed to write collection snapshot to {}",
                self.file_path.display()
            )
        })?;
        Ok(())
    }

    fn root(&self) -> &Path {
        &self.root
    }
}
