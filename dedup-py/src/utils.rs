use numpy::PyReadonlyArrayDyn;
use pyo3::{prelude::*, types::PyType};
use std::collections::HashMap;

use dedup_core::UnionFind as CoreUnionFind;

// Python wrapper for UnionFind
#[pyclass(name = "UnionFind")]
#[derive(Clone)]
pub struct UnionFind {
    pub inner: CoreUnionFind,
}

#[pymethods]
impl UnionFind {
    #[new]
    fn new() -> Self {
        UnionFind {
            inner: CoreUnionFind::new(),
        }
    }

    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let inner = CoreUnionFind::load(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load: {}", e))
        })?;
        Ok(UnionFind { inner })
    }

    fn find(&mut self, x: usize) -> usize {
        self.inner.find(x)
    }

    #[pyo3(name = "batch_find")]
    fn batch_find<'py>(
        mut slf: PyRefMut<'py, Self>,
        batched_idx: PyReadonlyArrayDyn<'py, u32>,
    ) -> Vec<usize> {
        let batched_idx = batched_idx.as_array();
        let results: Vec<usize> = batched_idx
            .iter()
            .map(|&x| slf.inner.find(x as usize))
            .collect();
        results
    }

    fn union(&mut self, x: usize, y: usize) {
        self.inner.union(x, y);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn dump(&self, path: &str) -> PyResult<()> {
        self.inner.dump(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to dump: {}", e))
        })
    }

    #[getter]
    fn parent(&self) -> HashMap<usize, usize> {
        self.inner.parent.clone()
    }

    #[getter]
    fn rank(&self) -> HashMap<usize, usize> {
        self.inner.rank.clone()
    }

    #[getter]
    fn edges(&self) -> usize {
        self.inner.edges
    }
}
