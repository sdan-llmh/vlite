//! Auto-chunking — recursive splitting + overlap + parent-child + contextual headers.
//!
//! The production-tested pattern: embed small children (~256 tokens), return big parents (~1024 tokens).
//! Prepend contextual headers to children before embedding (Anthropic's 49% improvement).

/// A chunk with its parent context.
#[derive(Debug, Clone)]
pub struct ChunkWithParent {
    /// The child text WITH contextual header prepended (this gets embedded).
    pub child_text: String,
    /// The parent text — larger surrounding context (this gets returned on search).
    pub parent_text: String,
    /// Byte offset of the child in the original text (before header).
    pub byte_offset: usize,
    /// Index of this child within its parent group.
    pub child_index: usize,
    /// Total children in the same parent group.
    pub children_in_parent: usize,
}

/// Separators in priority order: paragraphs → lines → sentences → words.
const SEPARATORS: &[&str] = &["\n\n", "\n", ". ", " "];

/// Recursively split `text` into pieces of at most `max_chars` characters,
/// trying the strongest separator first.
fn recursive_split<'a>(text: &'a str, max_chars: usize, sep_idx: usize) -> Vec<&'a str> {
    if text.len() <= max_chars || sep_idx >= SEPARATORS.len() {
        // Base case: fits, or no more separators — hard-cut
        if text.len() <= max_chars {
            return if text.is_empty() { vec![] } else { vec![text] };
        }
        // Hard cut at char boundary
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < text.len() {
            let end = (start + max_chars).min(text.len());
            // Don't split in the middle of a UTF-8 char
            let end = text.floor_char_boundary(end);
            if end <= start {
                break;
            }
            pieces.push(&text[start..end]);
            start = end;
        }
        return pieces;
    }

    let sep = SEPARATORS[sep_idx];
    let parts: Vec<&str> = text.split(sep).collect();

    if parts.len() <= 1 {
        // This separator doesn't split the text — try next
        return recursive_split(text, max_chars, sep_idx + 1);
    }

    // Merge small parts together, recurse on big ones
    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_start = 0usize;

    for (i, part) in parts.iter().enumerate() {
        let would_be = if current.is_empty() {
            part.len()
        } else {
            current.len() + sep.len() + part.len()
        };

        if would_be <= max_chars {
            if !current.is_empty() {
                current.push_str(sep);
            }
            current.push_str(part);
        } else {
            // Flush current if non-empty
            if !current.is_empty() {
                // Current fits — add it directly
                // Find the slice in original text
                let offset = byte_offset_of_substr(text, &current, current_start);
                let end = offset + current.len();
                result.push(&text[offset..end.min(text.len())]);
                current_start = end + sep.len();
                current.clear();
            }
            // If part itself is too big, recurse with next separator
            if part.len() > max_chars {
                let sub = recursive_split(part, max_chars, sep_idx + 1);
                result.extend(sub);
                current_start = byte_offset_of_substr(text, part, current_start) + part.len() + sep.len();
            } else {
                current.push_str(part);
            }
        }

        // Last part — flush
        if i == parts.len() - 1 && !current.is_empty() {
            let offset = byte_offset_of_substr(text, &current, current_start);
            let end = offset + current.len();
            result.push(&text[offset..end.min(text.len())]);
        }
    }

    result.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

/// Find byte offset of `needle` content starting search from `from`.
fn byte_offset_of_substr(haystack: &str, needle: &str, from: usize) -> usize {
    // Try to find the needle starting from `from`
    if let Some(pos) = haystack[from..].find(needle) {
        from + pos
    } else {
        from.min(haystack.len())
    }
}

/// Chunk text into children with parent context + contextual headers.
///
/// - `text`: the full document text
/// - `title`: document title for contextual headers (can be empty)
/// - `child_max_chars`: max chars per child (~256 tokens ≈ 1024 chars)
/// - `parent_max_chars`: max chars per parent (~1024 tokens ≈ 4096 chars)
/// - `overlap_chars`: overlap between adjacent children (~50 tokens ≈ 200 chars)
pub fn chunk(
    text: &str,
    title: &str,
    child_max_chars: usize,
    parent_max_chars: usize,
    overlap_chars: usize,
) -> Vec<ChunkWithParent> {
    if text.trim().is_empty() {
        return vec![];
    }

    // If text fits in a single child, return it as-is (parent = child)
    if text.len() <= child_max_chars {
        let header = if title.is_empty() {
            String::new()
        } else {
            format!("From {title}: ")
        };
        return vec![ChunkWithParent {
            child_text: format!("{header}{text}"),
            parent_text: text.to_string(),
            byte_offset: 0,
            child_index: 0,
            children_in_parent: 1,
        }];
    }

    // Step 1: Recursive split into raw children
    let raw_children = recursive_split(text, child_max_chars, 0);

    // Step 2: Apply overlap — prepend tail of previous chunk
    let mut children_with_offsets: Vec<(String, usize)> = Vec::with_capacity(raw_children.len());
    for (i, child) in raw_children.iter().enumerate() {
        let byte_offset = byte_offset_of_substr(text, child, 0);
        if i > 0 && overlap_chars > 0 {
            // Grab overlap from previous child's tail
            let prev = &raw_children[i - 1];
            let overlap_start = prev.len().saturating_sub(overlap_chars);
            let overlap_start = prev.ceil_char_boundary(overlap_start);
            let overlap_text = &prev[overlap_start..];
            let merged = format!("{overlap_text}{child}");
            children_with_offsets.push((merged, byte_offset.saturating_sub(overlap_text.len())));
        } else {
            children_with_offsets.push((child.to_string(), byte_offset));
        }
    }

    // Step 3: Group children into parents
    let mut results = Vec::new();
    let mut parent_start = 0usize;
    let mut group: Vec<(String, usize)> = Vec::new();
    let mut group_len = 0usize;

    let flush_group = |group: &[(String, usize)],
                       text: &str,
                       parent_start: usize,
                       title: &str,
                       results: &mut Vec<ChunkWithParent>| {
        if group.is_empty() {
            return;
        }
        // Parent = span from first child start to last child end
        let first_offset = group[0].1;
        let last = &group[group.len() - 1];
        let last_end = last.1 + last.0.len();
        let parent_end = last_end.min(text.len());
        let parent_begin = first_offset.min(parent_start);
        let parent_text = &text[parent_begin..parent_end];

        let header = if title.is_empty() {
            String::new()
        } else {
            format!("From {title}: ")
        };

        for (ci, (child, offset)) in group.iter().enumerate() {
            results.push(ChunkWithParent {
                child_text: format!("{header}{child}"),
                parent_text: parent_text.to_string(),
                byte_offset: *offset,
                child_index: ci,
                children_in_parent: group.len(),
            });
        }
    };

    for (child_text, offset) in &children_with_offsets {
        if group_len + child_text.len() > parent_max_chars && !group.is_empty() {
            flush_group(&group, text, parent_start, title, &mut results);
            parent_start = *offset;
            group.clear();
            group_len = 0;
        }
        group_len += child_text.len();
        group.push((child_text.clone(), *offset));
    }
    // Flush remaining
    flush_group(&group, text, parent_start, title, &mut results);

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_text() {
        let chunks = chunk("hello world", "doc", 1024, 4096, 200);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].child_text.starts_with("From doc: "));
        assert_eq!(chunks[0].parent_text, "hello world");
    }

    #[test]
    fn test_paragraph_splitting() {
        let text = "First paragraph about biology.\n\nSecond paragraph about physics.\n\nThird paragraph about chemistry.";
        let chunks = chunk(text, "", 40, 200, 0);
        assert!(chunks.len() >= 2, "should split on paragraph boundaries");
    }

    #[test]
    fn test_empty_text() {
        let chunks = chunk("", "doc", 1024, 4096, 200);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parent_child_grouping() {
        let text = (0..20)
            .map(|i| format!("Paragraph {i} with some content about topic {i}."))
            .collect::<Vec<_>>()
            .join("\n\n");
        let chunks = chunk(&text, "test", 60, 250, 0);
        // Parents should group multiple children
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.child_text.starts_with("From test: "));
            assert!(!c.parent_text.is_empty());
        }
    }
}
