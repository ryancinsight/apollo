// ── Canonical SSOT primitive root table ──────────────────────────────────────
// This file is the single source of truth for prime→generator mappings.
// Both `apollo-fft-macros` (proc-macro expansion) and `apollo-fft` (runtime)
// include it via `include!`, so the two crates always agree on the table.
//
// Each entry is `(prime, smallest_primitive_root_modulo_prime)`.

#[allow(dead_code)]
pub(crate) const PRIMITIVE_ROOTS: &[(usize, usize)] = &[
    (2, 1),
    (3, 2),
    (5, 2),
    (7, 3),
    (11, 2),
    (13, 2),
    (17, 3),
    (19, 2),
    (23, 5),
    (29, 2),
    (31, 3),
    (37, 2),
    (41, 6),
    (43, 3),
    (47, 5),
    (53, 2),
    (59, 2),
    (61, 2),
    (67, 2),
    (71, 7),
    (73, 5),
    (79, 3),
    (83, 2),
    (89, 3),
    (97, 5),
    (101, 2),
    (109, 6),
    (113, 3),
    (127, 3),
    (131, 2),
    (151, 6),
    (167, 5),
    (173, 2),
    (179, 2),
    (181, 2),
    (193, 5),
    (197, 2),
    (199, 3),
    (10007, 5),
];
