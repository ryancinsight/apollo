/// Whether a codelet emits separate transform phases for local experiments.
pub(crate) enum PhaseEmission {
    Omit,
    TestOnly,
}
