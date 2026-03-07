use std::collections::BTreeMap;

pub type Metadata = BTreeMap<String, String>;

pub fn matches_filter(metadata: &Metadata, filter: Option<&Metadata>) -> bool {
    filter.map_or(true, |required| {
        required
            .iter()
            .all(|(key, expected)| metadata.get(key) == Some(expected))
    })
}
