//! Sparse FFT plan configuration.

use apollo_fft::{ApolloError, ApolloResult};

/// How the plan finds the sparse support (ADR 0064).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RecoveryRoute {
    /// The dense `O(N log N)` transform ranked by a top-`K` heap: exact for
    /// every input, the oracle for the other route.
    #[default]
    DenseTopK,
    /// Aliasing onto `bucket_count` buckets by downsampling and syndrome
    /// decoding of each bucket, the count doubling on a failed round:
    /// `O(K log K)` while no bucket holds more than four tones, exact for any
    /// exactly `K`-sparse input.
    Downsampled,
}

/// Validated sparse FFT configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseFftConfig {
    n: usize,
    k: usize,
    bucket_count: usize,
    trials: usize,
    threshold: f64,
    route: RecoveryRoute,
}

impl SparseFftConfig {
    /// Create a validated sparse FFT configuration on the dense route.
    ///
    /// The bucket count is `min(next_power_of_two(4k), n)`, the aliasing
    /// model's starting count. The number of recovery rounds is
    /// `max(4, floor(log2(n)) + 1)`, which carries the downsampled route from
    /// that count to `n` by doubling.
    pub fn new(n: usize, k: usize) -> ApolloResult<Self> {
        if n == 0 {
            return Err(ApolloError::validation(
                "n",
                n.to_string(),
                "signal length must be non-zero",
            ));
        }
        if k == 0 {
            return Err(ApolloError::validation(
                "k",
                k.to_string(),
                "sparsity must be non-zero",
            ));
        }

        Ok(Self {
            n,
            k,
            bucket_count: (4 * k).next_power_of_two().min(n),
            trials: (n.ilog2() as usize + 1).max(4),
            threshold: 0.0,
            route: RecoveryRoute::DenseTopK,
        })
    }

    /// Create a validated configuration on the downsampled route (ADR 0064).
    ///
    /// The route aliases the signal onto `bucket_count` buckets and doubles
    /// the count on a failed round up to `n`, so every count on the way must
    /// divide `n`: `n` must be the starting count times a power of two.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::new`], and a validation error on a length the
    /// doubling cannot reach.
    pub fn downsampled(n: usize, k: usize) -> ApolloResult<Self> {
        let mut cfg = Self::new(n, k)?;
        let cofactor = n / cfg.bucket_count;
        if !n.is_multiple_of(cfg.bucket_count) || !cofactor.is_power_of_two() {
            return Err(ApolloError::validation(
                "n",
                n.to_string(),
                "the downsampled route needs a length that is its bucket count times a power of two",
            ));
        }
        cfg.route = RecoveryRoute::Downsampled;
        Ok(cfg)
    }

    /// Create a sparse FFT configuration with an explicit threshold.
    ///
    /// `n` is the signal length, `k` is the max coefficients to retain, and
    /// `threshold` is the minimum magnitude a coefficient must exceed to be
    /// retained (0.0 = keep top-K regardless, excluding exact zero).
    pub fn new_with_threshold(n: usize, k: usize, threshold: f64) -> ApolloResult<Self> {
        let mut cfg = Self::new(n, k)?;
        cfg.threshold = threshold;
        Ok(cfg)
    }
    /// Return the signal length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.n
    }

    /// Return whether the configured signal length is zero.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.n == 0
    }

    /// Return the target sparsity.
    #[must_use]
    pub const fn sparsity(self) -> usize {
        self.k
    }

    /// Return the aliasing bucket count.
    #[must_use]
    pub const fn bucket_count(self) -> usize {
        self.bucket_count
    }

    /// Return the number of deterministic recovery trials.
    #[must_use]
    pub const fn trials(self) -> usize {
        self.trials
    }

    /// Return the coefficient selection threshold.
    #[must_use]
    pub const fn threshold(self) -> f64 {
        self.threshold
    }

    /// Return the recovery route.
    #[must_use]
    pub const fn route(self) -> RecoveryRoute {
        self.route
    }
}
