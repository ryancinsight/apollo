//! Combinatorial Laplacian and graph spectral basis construction.

use crate::domain::graph::adjacency::GraphAdjacency;
use leto::{Array2, Storage};
use nalgebra::{DMatrix, SymmetricEigen};

/// Laplacian eigensystem stored for application-layer plans.
#[derive(Debug, Clone)]
pub struct GraphSpectralBasis {
    /// Laplacian eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Column-major eigenvector matrix.
    pub eigenvectors: Vec<f64>,
}

/// Build the combinatorial Laplacian `L = D - A`.
#[must_use]
pub fn combinatorial_laplacian(graph: &GraphAdjacency) -> Array2<f64> {
    let adjacency = graph.matrix();
    let n = graph.len();
    let mut values = vec![0.0; n * n];
    for row in 0..n {
        let degree: f64 = (0..n)
            .map(|col| *adjacency.get([row, col]).expect("validated adjacency"))
            .sum();
        values[row * n + row] = degree;
        for col in 0..n {
            values[row * n + col] -= *adjacency.get([row, col]).expect("validated adjacency");
        }
    }
    Array2::from_shape_vec([n, n], values).expect("laplacian shape must match storage")
}

/// Compute the graph Fourier basis from the combinatorial Laplacian.
#[must_use]
pub fn spectral_basis(graph: &GraphAdjacency) -> GraphSpectralBasis {
    let laplacian = combinatorial_laplacian(graph);
    let n = graph.len();
    let decomposition = SymmetricEigen::new(DMatrix::from_row_slice(
        n,
        n,
        laplacian.storage().as_slice(),
    ));
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&lhs, &rhs| {
        decomposition.eigenvalues[lhs]
            .partial_cmp(&decomposition.eigenvalues[rhs])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Vec::with_capacity(n * n);
    for &column in &order {
        eigenvalues.push(decomposition.eigenvalues[column]);
        for row in 0..n {
            eigenvectors.push(decomposition.eigenvectors[(row, column)]);
        }
    }

    GraphSpectralBasis {
        eigenvalues,
        eigenvectors,
    }
}
