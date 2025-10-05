use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, fs};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct UnionFind {
    pub parent: HashMap<usize, usize>,
    pub rank: HashMap<usize, usize>,
    #[serde(default)]
    pub edges: usize,
}

impl Default for UnionFind {
    fn default() -> Self {
        Self::new()
    }
}

impl UnionFind {
    // Constructor to create a new UnionFind instance
    pub fn new() -> Self {
        UnionFind {
            parent: HashMap::new(),
            rank: HashMap::new(),
            edges: 0,
        }
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let deserialized: UnionFind = serde_json::from_str(&content)?;
        Ok(deserialized)
    }

    // Find with path compression
    pub fn find(&mut self, x: usize) -> usize {
        // Temporarily take the value out of `self.parent` to avoid double mutable borrow.
        let parent_value = *self.parent.entry(x).or_insert(x);

        // Check if we need to recurse to compress the path.
        if parent_value == x {
            // If the parent is the root, return the root.
            parent_value
        } else {
            // If the parent is not the root, recurse to find the root.
            let root = self.find(parent_value);
            // Update the parent to the root.
            self.parent.insert(x, root);
            root
        }
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);

        // If both elements are already in the same set, do nothing
        if px == py {
            return;
        }

        self.edges += 1;
        let rank_px = *self.rank.entry(px).or_insert(0);
        let rank_py = *self.rank.entry(py).or_insert(0);

        match rank_px.cmp(&rank_py) {
            Ordering::Equal => {
                // If ranks are equal, make px the parent and increment its rank
                self.parent.insert(py, px);
                *self.rank.entry(px).or_insert(0) += 1;
            }
            Ordering::Greater => {
                // If px has higher rank, make it the parent
                self.parent.insert(py, px);
            }
            Ordering::Less => {
                // If py has higher rank, make it the parent
                self.parent.insert(px, py);
            }
        }
    }

    pub fn reset(&mut self) {
        self.parent.clear();
        self.rank.clear();
        self.edges = 0;
    }

    pub fn dump(&self, path: &str) -> Result<(), std::io::Error> {
        let serialized = serde_json::to_string(&self).unwrap();
        std::fs::write(path, serialized)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_union_find_operations() {
        let start_process = Instant::now();
        let mut uf = UnionFind::new();
        uf.union(1, 2);
        uf.union(2, 3);
        uf.union(4, 5);
        assert_eq!(uf.find(1), 1);
        assert_eq!(uf.find(2), 1);
        assert_eq!(uf.find(3), 1);
        assert_eq!(uf.find(4), 4);
        assert_eq!(uf.find(5), 4);
        assert_eq!(*uf.rank.get(&1).unwrap(), 1);
        assert_eq!(*uf.rank.get(&2).unwrap(), 0);
        uf.union(3, 4);
        assert!(uf.find(1) == uf.find(5));
        assert_eq!(uf.find(7), 7);
        assert_eq!(*uf.rank.get(&7).unwrap_or(&0), 0);
        let process_duration = start_process.elapsed();

        dbg!(process_duration);

        // try saving
        uf.dump("union_find_test.json").unwrap();

        // Clean up test file
        let _ = std::fs::remove_file("union_find_test.json");
    }
}
