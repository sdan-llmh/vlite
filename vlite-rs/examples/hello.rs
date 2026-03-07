//! Hello world example for vlite-rs.
//!
//! ```bash
//! cargo run --example hello
//! ```
//!
//! Downloads the default model (~80MB) on first run.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating vlite database...");
    let mut db = vlite::VLite::new()?;

    println!("Adding documents...");
    db.add("the mitochondria is the powerhouse of the cell", None)?;
    db.add("photosynthesis converts sunlight into chemical energy", None)?;
    db.add("the earth orbits the sun in approximately 365 days", None)?;
    db.add("machine learning is a subset of artificial intelligence", None)?;
    db.add("error code XJ-7482: connection timeout on port 443", None)?;

    println!("Database has {} items\n", db.len());

    // Semantic search
    println!("=== Search: 'biology energy' ===");
    let results = db.search("biology energy", 3)?;
    for r in &results {
        println!("  {:.4} | {}", r.score, r.text);
    }

    // Keyword-sensitive search (hybrid catches exact matches)
    println!("\n=== Search: 'error XJ-7482' ===");
    let results = db.search("error XJ-7482", 3)?;
    for r in &results {
        println!("  {:.4} | {}", r.score, r.text);
    }

    // Persistence
    let path = "hello.vlite";
    db.save(path)?;
    println!("\nSaved to {path}");

    let db2 = vlite::VLite::open(path)?;
    println!("Reopened: {} items", db2.len());

    // Cleanup
    std::fs::remove_file(path).ok();

    Ok(())
}
