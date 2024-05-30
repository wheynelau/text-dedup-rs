use std::collections::BTreeMap;

struct UnionFind {
    parent: BTreeMap<usize, usize>,
    rank: BTreeMap<usize, usize>,
}

impl UnionFind {
    // Constructor to create a new UnionFind instance
    pub fn new() -> Self {
        UnionFind {
            parent: BTreeMap::new(),
            rank: BTreeMap::new(),
        }
    }

    // Find with path compression
    pub fn find(&mut self, x: usize) -> usize {
        // Temporarily take the value out of `self.parent` to avoid double mutable borrow.
        let parent_value = *self.parent.entry(x).or_insert(x);
    
        // Check if we need to recurse to compress the path.
        if parent_value != x {
            let root = self.find(parent_value);
            // Re-insert the value with the updated root after the mutable borrow ends.
            *self.parent.entry(x).or_insert(x) = root;
        }
    
        // Return the final root, which is now guaranteed to be correct.
        self.parent[&x]
    }

    // Union by rank
    pub fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            let rank_x = *self.rank.entry(root_x).or_insert(0);
            let rank_y = *self.rank.entry(root_y).or_insert(0);

            if rank_x > rank_y {
                self.parent.insert(root_y, root_x);
            } else if rank_x < rank_y {
                self.parent.insert(root_x, root_y);
            } else {
                self.parent.insert(root_y, root_x);
                *self.rank.entry(root_x).or_insert(0) += 1;
            }
        }
    }

    pub fn reset(&mut self) {
        self.parent.clear();
        self.rank.clear();
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
        assert_eq!(*uf.rank.get(&7).unwrap_or(&0) , 0);
        let process_duration = start_process.elapsed();

        dbg!(process_duration);
    }
}