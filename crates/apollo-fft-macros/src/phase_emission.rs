use proc_macro2::TokenStream;
use quote::quote;

/// Whether generated phases inline directly or admit a static schedule.
pub(crate) enum PhaseEmission {
    Inline,
    Scheduled,
}

impl PhaseEmission {
    pub(crate) fn schedule_parameter(&self) -> TokenStream {
        match self {
            Self::Inline => quote! {},
            Self::Scheduled => quote! {
                S: crate::application::execution::kernel::components::winograd::composite::schedule::Schedule,
            },
        }
    }

    pub(crate) fn phase(&self, body: TokenStream) -> TokenStream {
        match self {
            Self::Inline => body,
            Self::Scheduled => quote! {
                {
                    #[inline(never)]
                    fn run_phase(operation: impl ::core::ops::FnOnce()) {
                        operation();
                    }

                    let operation = || { #body };
                    if S::SPLIT_PHASES {
                        run_phase(operation);
                    } else {
                        ({ operation })();
                    }
                }
            },
        }
    }
}
