//! Extraction helpers — PDF text extraction + HTML stripping.
//!
//! These are standalone functions, not methods on VLite.
//! User calls them to get text, then passes to db.add().

use crate::error::{Result, VLiteError};
use std::path::Path;

/// Extract all text from a PDF file.
///
/// ```no_run
/// let text = vlite::extract_pdf("paper.pdf").unwrap();
/// ```
pub fn extract_pdf(path: impl AsRef<Path>) -> Result<String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| VLiteError::Io(format!("cannot read PDF: {e}")))?;
    pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| VLiteError::Io(format!("PDF extraction failed: {e}")))
}

/// Strip HTML tags and decode entities → plain text.
///
/// Simple implementation: strips everything between < and >.
/// Decodes common entities (&amp; &lt; &gt; &quot; &#39; &nbsp;).
/// Collapses whitespace.
pub fn extract_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;

    let lower = html.to_lowercase();
    let bytes = html.as_bytes();
    let lower_bytes = lower.as_bytes();

    let mut i = 0;
    while i < bytes.len() {
        if !in_tag && bytes[i] == b'<' {
            in_tag = true;
            // Check for script/style open tags
            if lower_bytes[i..].starts_with(b"<script") {
                in_script = true;
            } else if lower_bytes[i..].starts_with(b"<style") {
                in_style = true;
            }
            // Check for close tags
            if lower_bytes[i..].starts_with(b"</script") {
                in_script = false;
            } else if lower_bytes[i..].starts_with(b"</style") {
                in_style = false;
            }
            i += 1;
            continue;
        }
        if in_tag {
            if bytes[i] == b'>' {
                in_tag = false;
                result.push(' ');
            }
            i += 1;
            continue;
        }
        if in_script || in_style {
            i += 1;
            continue;
        }
        // Handle entities
        if bytes[i] == b'&' {
            if lower_bytes[i..].starts_with(b"&amp;") {
                result.push('&');
                i += 5;
            } else if lower_bytes[i..].starts_with(b"&lt;") {
                result.push('<');
                i += 4;
            } else if lower_bytes[i..].starts_with(b"&gt;") {
                result.push('>');
                i += 4;
            } else if lower_bytes[i..].starts_with(b"&quot;") {
                result.push('"');
                i += 6;
            } else if lower_bytes[i..].starts_with(b"&#39;") || lower_bytes[i..].starts_with(b"&apos;") {
                result.push('\'');
                i += if lower_bytes[i..].starts_with(b"&#39;") { 5 } else { 6 };
            } else if lower_bytes[i..].starts_with(b"&nbsp;") {
                result.push(' ');
                i += 6;
            } else {
                result.push('&');
                i += 1;
            }
            continue;
        }
        result.push(bytes[i] as char);
        i += 1;
    }

    // Collapse whitespace
    let mut collapsed = String::with_capacity(result.len());
    let mut prev_space = false;
    for ch in result.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                collapsed.push(' ');
                prev_space = true;
            }
        } else {
            collapsed.push(ch);
            prev_space = false;
        }
    }

    collapsed.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_html() {
        let html = "<html><head><title>Test</title></head><body><h1>Hello</h1><p>World &amp; friends</p></body></html>";
        let text = extract_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World & friends"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_extract_html_script() {
        let html = "<p>Before</p><script>var x = 1;</script><p>After</p>";
        let text = extract_html(html);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("var x"));
    }
}
