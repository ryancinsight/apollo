use super::{emit_codelets, WinogradCompositesInput};
use quote::{quote, ToTokens};

const PAIRS: &str = "gt_pairs: [(2, 3), (2, 25)], ct_pairs: [(5, 5), (12, 12)], pp_pairs: []";

fn codelets(specification: &str) -> Vec<syn::ItemFn> {
    let input: WinogradCompositesInput =
        syn::parse_str(specification).expect("valid specification");
    let file: syn::File = syn::parse2(emit_codelets(&input)).expect("generated Rust parses");
    file.items
        .into_iter()
        .map(|item| match item {
            syn::Item::Fn(function) => function,
            _ => panic!("codelet expansion must contain only the requested functions"),
        })
        .collect()
}

#[test]
fn ordinary_pairs_keep_scalar_and_direction_signature() {
    let functions = codelets(PAIRS);
    let names: Vec<_> = functions
        .iter()
        .map(|function| function.sig.ident.to_string())
        .collect();
    assert_eq!(
        names,
        ["dft6_impl", "dft50_impl", "dft25_impl", "dft144_impl"]
    );
    for function in functions {
        let parameters: Vec<_> = function
            .sig
            .generics
            .params
            .iter()
            .map(|parameter| match parameter {
                syn::GenericParam::Type(parameter) => parameter.ident.to_string(),
                syn::GenericParam::Const(parameter) => parameter.ident.to_string(),
                syn::GenericParam::Lifetime(_) => panic!("unexpected lifetime parameter"),
            })
            .collect();
        assert_eq!(parameters, ["F", "INVERSE"]);
        assert!(!function.to_token_stream().to_string().contains("schedule"));
    }
}

#[test]
fn scheduling_changes_only_explicitly_selected_pairs() {
    for (selection, scheduled_names) in [
        ("[(2, 25)]", vec!["dft50_impl"]),
        ("[(12, 12)]", vec!["dft144_impl"]),
        ("[(2, 25), (12, 12)]", vec!["dft50_impl", "dft144_impl"]),
    ] {
        let functions = codelets(&format!("{PAIRS}, scheduled_pairs: {selection}"));
        assert_eq!(functions.len(), 4, "each requested codelet has one body");
        for function in functions {
            let scheduled = scheduled_names.contains(&function.sig.ident.to_string().as_str());
            assert_eq!(
                function.sig.generics.params.len(),
                if scheduled { 3 } else { 2 }
            );
            let body = function.block.to_token_stream().to_string();
            assert_eq!(
                body.matches("S :: SPLIT_PHASES").count(),
                if scheduled { 2 } else { 0 }
            );
            if scheduled {
                let parameter = function
                    .sig
                    .generics
                    .params
                    .last()
                    .expect("schedule parameter");
                assert_eq!(
                    parameter.to_token_stream().to_string(),
                    "S : crate :: application :: execution :: kernel :: components :: winograd :: composite :: schedule :: Schedule"
                );
            }
        }
    }
}

#[test]
fn generated_phase_owns_callback_invocation() {
    let arithmetic = quote! { output[0] = input[0]; };
    let emitted = crate::phase_emission::PhaseEmission::Scheduled.phase(arithmetic.clone());
    let phase: syn::ExprBlock = syn::parse2(emitted.clone()).expect("phase is a scoped block");
    let [syn::Stmt::Item(syn::Item::Fn(executor)), syn::Stmt::Local(operation), syn::Stmt::Expr(syn::Expr::If(branch), None)] =
        phase.block.stmts.as_slice()
    else {
        panic!("phase must define an executor, one closure, and its invocation branch");
    };
    assert_eq!(
        executor.to_token_stream().to_string(),
        quote! {
            #[inline(never)]
            fn run_phase(operation: impl ::core::ops::FnOnce()) {
                operation();
            }
        }
        .to_string()
    );
    assert_eq!(operation.pat.to_token_stream().to_string(), "operation");
    let initializer = operation
        .init
        .as_ref()
        .expect("phase closure is initialized");
    let syn::Expr::Closure(closure) = initializer.expr.as_ref() else {
        panic!("arithmetic must occur in one closure");
    };
    assert_eq!(
        closure.body.to_token_stream().to_string(),
        quote! { { #arithmetic } }.to_string()
    );
    assert_eq!(
        branch.cond.to_token_stream().to_string(),
        "S :: SPLIT_PHASES"
    );
    assert_eq!(
        branch.then_branch.to_token_stream().to_string(),
        quote! { { run_phase(operation); } }.to_string()
    );
    let (_, fused) = branch
        .else_branch
        .as_ref()
        .expect("fused branch is required");
    assert_eq!(
        fused.to_token_stream().to_string(),
        quote! { { ({ operation })(); } }.to_string()
    );
    assert_eq!(emitted.to_string().matches("S ::").count(), 1);
}

#[test]
fn empty_schedule_preserves_default_emission() {
    let ordinary: WinogradCompositesInput = syn::parse_str(PAIRS).expect("valid pairs");
    let explicit: WinogradCompositesInput =
        syn::parse_str(&format!("{PAIRS}, scheduled_pairs: []")).expect("valid empty schedule");
    assert_eq!(
        emit_codelets(&ordinary).to_string(),
        emit_codelets(&explicit).to_string()
    );
}

#[test]
fn schedule_rejects_absent_duplicate_or_wrong_family_pairs() {
    for (specification, diagnostic) in [
        (
            "scheduled_pairs: [(2, 25)]",
            "scheduled pair must occur exactly once in its transform pair list",
        ),
        (
            "ct_pairs: [(2, 25)], scheduled_pairs: [(2, 25)]",
            "scheduled pair must occur exactly once in its transform pair list",
        ),
        (
            "gt_pairs: [(12, 12)], scheduled_pairs: [(12, 12)]",
            "scheduled pair must occur exactly once in its transform pair list",
        ),
        (
            "gt_pairs: [(2, 25), (2, 25)], scheduled_pairs: [(2, 25)]",
            "scheduled pair must occur exactly once in its transform pair list",
        ),
        (
            "gt_pairs: [(2, 25)], scheduled_pairs: [(2, 25), (2, 25)]",
            "duplicate scheduled pair",
        ),
        (
            "gt_pairs: [(2, 3)], scheduled_pairs: [(2, 3)]",
            "scheduled pairs must be Good-Thomas (2, 25) or Cooley-Tukey (12, 12)",
        ),
        (
            "scheduled_pairs: [], scheduled_pairs: []",
            "duplicate `scheduled_pairs`",
        ),
    ] {
        let error = syn::parse_str::<WinogradCompositesInput>(specification)
            .err()
            .expect("invalid schedule must fail before emission");
        assert_eq!(error.to_string(), diagnostic);
    }
}
