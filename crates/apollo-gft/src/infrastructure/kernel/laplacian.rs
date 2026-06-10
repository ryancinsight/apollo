//! Combinatorial Laplacian and graph spectral basis construction.

use crate::domain::graph::adjacency::GraphAdjacency;
use leto::Array2;
use leto_ops::symmetric_eigen_jacobi;

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
    let decomposition = symmetric_eigen_jacobi(&laplacian.view())
        .expect("combinatorial Laplacian from validated undirected graph must be finite symmetric");

    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = Vec::with_capacity(n * n);
    for column in 0..n {
        eigenvalues.push(decomposition.eigenvalues[column]);
        for row in 0..n {
            eigenvectors.push(
                *decomposition
                    .eigenvectors
                    .get([row, column])
                    .expect("eigenvector matrix shape must match graph order"),
            );
        }
    }

    GraphSpectralBasis {
        eigenvalues,
        eigenvectors,
    }
}
