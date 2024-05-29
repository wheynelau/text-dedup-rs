use std::BTreeSet;

struct Union {
    parent: BTreeSet<i32>,
    rank: i32
}

impl Union {

    [#new]
    fn new() -> Self {
        Union {
            parent: BTreeSet::new(),
            rank: 0
        }
    }
}