use std::{cmp::Ordering, collections::HashMap, fs};
use pyo3::{prelude::*, types::PyType};
use serde::{Deserialize,Serialize};

#[pyclass]
#[derive(Clone,Serialize, Deserialize,Debug)]
pub struct UnionFind {
    #[pyo3(get)]
    parent: HashMap<usize, usize>,
    #[pyo3(get)]
    rank: HashMap<usize, usize>,
}

#[pymethods]
impl UnionFind {
    // Constructor to create a new UnionFind instance
    #[new]
    pub fn new() -> Self {
        UnionFind {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    #[classmethod]
    pub fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let content = fs::read_to_string(path).expect("Unable to read file");
        let deserialized: UnionFind = serde_json::from_str(&content).unwrap();
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
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            let rank_x = *self.rank.entry(root_x).or_insert(0);
            let rank_y = *self.rank.entry(root_y).or_insert(0);

            match rank_x.cmp(&rank_y) {
                Ordering::Greater => {
                    // If rank_x is greater than rank_y, make root_x the parent of root_y
                    self.parent.insert(root_y, root_x);
                },
                Ordering::Less => {
                    // If rank_x is less than rank_y, make root_y the parent of root_x
                    self.parent.insert(root_x, root_y);
                },
                Ordering::Equal => {
                    // If ranks are equal, make root_x the parent of root_y and increment the rank of root_x
                    self.parent.insert(root_y, root_x);
                    *self.rank.entry(root_x).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.parent.clear();
        self.rank.clear();
    }

    pub fn dump(&self, path: &str) -> PyResult<()> {
        let serialized = serde_json::to_string(&self).unwrap();
        std::fs::write(path, serialized).expect("Unable to write to file");
        Ok(())
    }
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_union_find_operations() {
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
        assert_eq!(*uf.rank.get(&7).unwrap_or(&0) , 0);
        // try saving
        uf.dump("union_find.json").unwrap();

    }
}