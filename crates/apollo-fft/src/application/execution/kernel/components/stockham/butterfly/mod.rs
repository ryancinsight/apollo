#[cfg(target_arch = "x86_64")]
pub(crate) mod dispatch;
#[cfg(target_arch = "x86_64")]
pub(crate) mod fixed;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) mod hybrid;
pub(crate) mod stage;

#[cfg(target_arch = "x86_64")]
pub(crate) use dispatch::*;
#[cfg(target_arch = "x86_64")]
pub(crate) use fixed::*;
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) use hybrid::*;
pub(crate) use stage::*;
