use crate::document::{Document, SegmentKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkDraft {
    pub kind: SegmentKind,
    pub path: Vec<String>,
    pub text: String,
}

pub trait Chunker {
    fn chunk_document(&self, document: &Document) -> Vec<ChunkDraft>;
}

#[derive(Debug, Default, Clone)]
pub struct StructuralChunker;

impl StructuralChunker {
    fn flush_buffer(
        drafts: &mut Vec<ChunkDraft>,
        headings: &[String],
        paragraph_lines: &mut Vec<String>,
    ) {
        if paragraph_lines.is_empty() {
            return;
        }

        let text = paragraph_lines.join(" ").trim().to_string();
        paragraph_lines.clear();

        if text.is_empty() {
            return;
        }

        drafts.push(ChunkDraft {
            kind: SegmentKind::Paragraph,
            path: headings.to_vec(),
            text,
        });
    }
}

impl Chunker for StructuralChunker {
    fn chunk_document(&self, document: &Document) -> Vec<ChunkDraft> {
        let mut drafts = Vec::new();
        let mut headings: Vec<String> = Vec::new();
        let mut paragraph_lines: Vec<String> = Vec::new();

        for raw_line in document.raw_text.lines() {
            let line = raw_line.trim();

            if line.is_empty() {
                Self::flush_buffer(&mut drafts, &headings, &mut paragraph_lines);
                continue;
            }

            if let Some(stripped) = line.strip_prefix('#') {
                Self::flush_buffer(&mut drafts, &headings, &mut paragraph_lines);

                let level = line.chars().take_while(|c| *c == '#').count();
                let title = stripped.trim_start_matches('#').trim().to_string();
                if title.is_empty() {
                    continue;
                }

                if headings.len() >= level {
                    headings.truncate(level.saturating_sub(1));
                }
                headings.push(title);
                continue;
            }

            paragraph_lines.push(line.to_string());
        }

        Self::flush_buffer(&mut drafts, &headings, &mut paragraph_lines);

        if drafts.is_empty() && !document.raw_text.trim().is_empty() {
            drafts.push(ChunkDraft {
                kind: SegmentKind::Paragraph,
                path: Vec::new(),
                text: document.raw_text.trim().to_string(),
            });
        }

        drafts
    }
}

#[cfg(test)]
mod tests {
    use super::{Chunker, StructuralChunker};
    use crate::document::Document;
    use crate::metadata::Metadata;

    #[test]
    fn chunks_markdown_by_heading_and_paragraph() {
        let doc = Document {
            id: "doc-1".into(),
            title: Some("Example".into()),
            raw_text: "# Intro\nHello world.\n\n# Details\nSecond paragraph.".into(),
            metadata: Metadata::new(),
        };

        let chunker = StructuralChunker;
        let chunks = chunker.chunk_document(&doc);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].path, vec!["Intro".to_string()]);
        assert_eq!(chunks[0].text, "Hello world.");
        assert_eq!(chunks[1].path, vec!["Details".to_string()]);
        assert_eq!(chunks[1].text, "Second paragraph.");
    }
}
