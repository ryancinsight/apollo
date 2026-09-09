mod private {
    pub trait Sealed {}

    impl Sealed for super::Fused {}

    #[cfg(test)]
    impl Sealed for super::Split {}
}

/// Chooses the inlining boundary of each generated transform phase.
///
/// Phase closures preserve arithmetic order and borrow caller-owned scratch.
/// The scalar and transform direction remain parameters of the codelet.
/// Generated code owns phase invocation and scratch initialization; the policy
/// selects only whether each phase crosses an outlined call boundary.
pub(crate) trait Schedule: private::Sealed {
    /// Whether phases execute through a non-inlined function.
    const SPLIT_PHASES: bool;
}

/// Production schedule: both phases inline into the codelet.
pub(crate) struct Fused;

impl Schedule for Fused {
    const SPLIT_PHASES: bool = false;
}

/// Experimental schedule: each phase has a separate call boundary.
#[cfg(test)]
pub(crate) struct Split;

#[cfg(test)]
impl Schedule for Split {
    const SPLIT_PHASES: bool = true;
}
