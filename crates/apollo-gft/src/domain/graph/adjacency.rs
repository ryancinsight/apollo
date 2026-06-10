//! Validated undirected weighted graph adjacency matrices.
//!
//! A graph Fourier transform over an undirected weighted graph uses the
//! eigensystem of the combinatorial Laplacian `L = D - A`. This descriptor is
//! the single validation boundary for matrix shape, symmetry, and finite edge
//! weights.

use crate::domain::contracts::error::{GftError, GftResult};
use leto::{Array2, ArrayView2, Storage};

const SYMMETRY_TOLERANCE: f64 = 1.0e-12;

/// Validated adjacency matrix for an undirected weighted graph.
#[derive(Debug, Clone)]
pub struct GraphAdjacency {
    matrix: Array2<f64>,
}

impl GraphAdjacency {
    /// Validate and store an undirected weighted adjacency matrix.
    pub fn new(matrix: Array2<f64>) -> GftResult<Self> {
        validate_adjacency(&matrix.view())?;
        Ok(Self { matrix })
    }

    /// Validate and copy an undirected weighted adjacency view.
    pub fn from_view(matrix: ArrayView2<'_, f64>) -> GftResult<Self> {
        validate_adjacency(&matrix)?;
        let [rows, cols] = matrix.shape();
        let mut values = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                values.push(*matrix.get([row, col]).expect("validated adjacency bounds"));
            }
        }
        let matrix = Array2::from_shape_vec([rows, cols], values).expect("validated shape");
        Ok(Self { matrix })
    }

    /// Return the graph order.
    #[must_use]
    pub fn len(&self) -> usize {
        self.matrix.shape()[0]
    }

    /// Return true when the graph order is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the validated adjacency matrix.
    #[must_use]
    pub fn matrix(&self) -> &Array2<f64> {
        &self.matrix
    }

    /// Return the C-contiguous backing values for the validated adjacency matrix.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.matrix.storage().as_slice()
    }
}

fn validate_adjacency(matrix: &ArrayView2<'_, f64>) -> GftResult<()> {
    let [rows, cols] = matrix.shape();
    if rows == 0 {
        return Err(GftError::EmptyGraph);
    }
    if rows != cols {
        return Err(GftError::NonSquareAdjacency);
    }
    for row in 0..rows {
        for col in 0..cols {
            let value = *matrix.get([row, col]).expect("validated adjacency bounds");
            if !value.is_finite() {
                return Err(GftError::NonFiniteWeight);
            }
            let transposed = *matrix
                .get([col, row])
                .expect("validated transposed adjacency bounds");
            if (value - transposed).abs() > SYMMETRY_TOLERANCE {
                return Err(GftError::NonSymmetricAdjacency);
            }
        }
    }
    Ok(())
}
