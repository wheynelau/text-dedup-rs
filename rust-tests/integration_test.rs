use dedup_rs::*;

#[test]
fn test_union_find_basic() {
    let mut uf = UnionFind::new();

    // Test basic union and find operations
    uf.union(0, 1);
    uf.union(2, 3);

    assert_eq!(uf.find(0), uf.find(1));
    assert_eq!(uf.find(2), uf.find(3));
    assert_ne!(uf.find(0), uf.find(2));

    // Union the two groups
    uf.union(1, 2);
    assert_eq!(uf.find(0), uf.find(3));
}

#[test]
fn test_union_find_transitive() {
    let mut uf = UnionFind::new();

    // Create a chain: 0-1-2-3-4
    uf.union(0, 1);
    uf.union(1, 2);
    uf.union(2, 3);
    uf.union(3, 4);

    // All should be in the same cluster
    let root = uf.find(0);
    assert_eq!(uf.find(1), root);
    assert_eq!(uf.find(2), root);
    assert_eq!(uf.find(3), root);
    assert_eq!(uf.find(4), root);
}

#[test]
fn test_union_find_multiple_clusters() {
    let mut uf = UnionFind::new();

    // Create multiple separate clusters
    uf.union(0, 1);
    uf.union(1, 2);  // Cluster 1: {0, 1, 2}

    uf.union(3, 4);
    uf.union(4, 5);  // Cluster 2: {3, 4, 5}

    uf.union(6, 7);  // Cluster 3: {6, 7}

    // Within cluster checks
    assert_eq!(uf.find(0), uf.find(2));
    assert_eq!(uf.find(3), uf.find(5));
    assert_eq!(uf.find(6), uf.find(7));

    // Between cluster checks
    assert_ne!(uf.find(0), uf.find(3));
    assert_ne!(uf.find(0), uf.find(6));
    assert_ne!(uf.find(3), uf.find(6));
}

#[test]
fn test_union_find_idempotent() {
    let mut uf = UnionFind::new();

    // Multiple unions of the same pair should be idempotent
    uf.union(0, 1);
    let root1 = uf.find(0);

    uf.union(0, 1);
    uf.union(1, 0);
    let root2 = uf.find(0);

    assert_eq!(root1, root2);
    assert_eq!(uf.find(0), uf.find(1));
}

// Note: EmbedFunc and embed_func tests would require Python initialization
// and access to PyO3 functions. Those are better tested through Python integration tests.
// The unit tests in src/embed.rs cover the core embedding logic.
