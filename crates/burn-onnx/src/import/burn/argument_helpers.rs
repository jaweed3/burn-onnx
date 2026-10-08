//! Helper functions for working with onnx_ir::Argument types
//!
//! This module provides utilities to generate code for different argument types
//! without needing the Type abstraction layer.

use onnx_ir::{
    Argument,
    ir::{ArgType, DType},
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;

use crate::burn::ToTokens;

/// Get the type TokenStream for a tensor rank and DType.
pub fn tensor_type_tokens(rank: usize, dtype: &DType) -> TokenStream {
    let rank = rank.to_tokens();
    match dtype {
        dtype if dtype.is_float() => quote! { Tensor<#rank> },
        dtype if dtype.is_int() || dtype.is_uint() => quote! { Tensor<#rank, Int> },
        dtype if dtype.is_bool() => quote! { Tensor<#rank, Bool> },
        _ => panic!("Unsupported tensor dtype: {:?}", dtype),
    }
}

/// The `DType` an argument is declared with, for use in a doc comment.
///
/// The Rust signature can't always carry the dtype (`Tensor<2>` is `Tensor<2, Int>` or
/// `Tensor<2>` depending on the element type), so `forward` documents it instead. This keeps
/// generated models honest about the ONNX types they were compiled from, matching the
/// "always specify explicit dtypes" convention the rest of the codegen already follows for
/// constants and boundary scalars.
pub fn arg_dtype_doc(arg: &Argument) -> String {
    let dtype = match &arg.ty {
        ArgType::Tensor(tensor) => &tensor.dtype,
        ArgType::ScalarNative(dtype) | ArgType::ScalarTensor(dtype) => dtype,
        // A shape input is a host-side `[i64; N]`; it has no element dtype to pin.
        ArgType::Shape(_) => return "shape ([i64; N])".into(),
    };

    match dtype {
        DType::Bool(_) => "bool".into(),
        other => format!("{other:?}"),
    }
}

/// `/// # Arguments` doc lines for a generated `forward`, one per argument.
///
/// Returns `None` when there is nothing to document, so the caller can omit the section
/// entirely rather than emit an empty header.
pub fn codegen_args_doc(args: &[Argument]) -> Option<TokenStream> {
    if args.is_empty() {
        return None;
    }

    let lines = args.iter().map(|arg| {
        // Use the ONNX name, not `arg_ident`: the parameter the user sees in the
        // signature is untagged, and the doc has to name the same thing.
        let name = &arg.name;
        let dtype = arg_dtype_doc(arg);
        let doc = format!(" `{name}`: expected dtype `{dtype}`");
        quote! { #[doc = #doc] }
    });

    Some(quote! {
        /// # Arguments
        #(#lines)*
    })
}

/// Get the type TokenStream for an argument
pub fn arg_type_tokens(arg: &Argument) -> TokenStream {
    match &arg.ty {
        ArgType::Tensor(tensor) => tensor_type_tokens(tensor.rank, &tensor.dtype),
        ArgType::ScalarNative(dtype) => scalar_type_tokens(dtype),
        ArgType::ScalarTensor(dtype) => match dtype {
            d if d.is_float() => quote! { Tensor<1> },
            d if d.is_int() || d.is_uint() => quote! { Tensor<1, Int> },
            d if d.is_bool() => quote! { Tensor<1, Bool> },
            _ => panic!("Unsupported scalar tensor dtype: {:?}", dtype),
        },
        ArgType::Shape(rank) => {
            let rank_lit = rank.to_tokens();
            quote! { [i64; #rank_lit] }
        }
    }
}

/// Get the type TokenStream for a scalar DType
pub fn scalar_type_tokens(dtype: &DType) -> TokenStream {
    match dtype {
        DType::F16 => quote! { half::f16 },
        DType::BF16 => quote! { half::bf16 },
        DType::F32 => quote! { f32 },
        DType::F64 => quote! { f64 },
        DType::I8 => quote! { i8 },
        DType::I16 => quote! { i16 },
        DType::I32 => quote! { i32 },
        DType::I64 => quote! { i64 },
        DType::U8 => quote! { u8 },
        DType::U16 => quote! { u16 },
        DType::U32 => quote! { u32 },
        DType::U64 => quote! { u64 },
        DType::Bool(_) => quote! { bool },
        _ => panic!("Unsupported scalar dtype: {:?}", dtype),
    }
}

/// Generate the `.into_scalar::<T>()` call fragment for extracting a native
/// scalar from a tensor.
///
/// Since burn made `Tensor::into_scalar` generic over the element type
/// (`fn into_scalar<E: Element>(self) -> E`), the element type must be supplied
/// at the call site rather than via a chained `.elem::<T>()`.
pub fn elem_cast_tokens(dtype: &DType) -> TokenStream {
    let ty = scalar_type_tokens(dtype);
    quote! { .into_scalar::<#ty>() }
}

/// Generate code to extract a native scalar from a on-device tensor.
///
/// Produces: `<input>.into_scalar::<T>()`
pub fn on_device_to_native(input: TokenStream, dtype: &DType) -> TokenStream {
    let cast = elem_cast_tokens(dtype);
    quote! { (#input)#cast }
}

/// Native `i64` expression for a scalar input, for the shape-arithmetic paths
/// where every value is converted to i64. A `ScalarTensor` lives on device, so
/// it is read back rather than named directly.
pub fn scalar_as_i64(arg: &Argument, value: TokenStream) -> TokenStream {
    let value = match &arg.ty {
        ArgType::ScalarTensor(dtype) => on_device_to_native(value, dtype),
        _ => value,
    };
    if arg.ty.elem_type() == DType::I64 {
        value
    } else {
        quote! { #value as i64 }
    }
}

/// Generate code to convert a ScalarTensor (Tensor<1>) to a Shape([i64; 1]).
///
/// Produces: `{ let v: T = <input>.into_scalar::<T>(); [v as i64] }`
pub fn scalar_tensor_to_shape(input: TokenStream, dtype: &DType) -> TokenStream {
    let ty = scalar_type_tokens(dtype);
    quote! {
        {
            let v: #ty = #input.into_scalar::<#ty>();
            [v as i64]
        }
    }
}

/// Generate code to convert a native scalar to a Shape([i64; 1]).
///
/// Produces: `[<input> as i64]`
pub fn scalar_native_to_shape(input: TokenStream) -> TokenStream {
    quote! { [#input as i64] }
}

/// Generate code to extract the first element of a Shape as a native scalar.
///
/// Produces: `<input>[0] as T`
pub fn shape_to_native(input: TokenStream, dtype: &DType) -> TokenStream {
    let ty = scalar_type_tokens(dtype);
    quote! { #input[0] as #ty }
}

/// The tagged identifier for `arg` (see `shadow_check::value_ident`); splice
/// it into tokens only, never stringify it.
pub fn arg_ident(arg: &Argument) -> Ident {
    super::shadow_check::value_ident(&arg.name)
}

/// Generate function parameters from a slice of arguments
///
/// Produces: `name1: Type1, name2: Type2, ...`
pub fn codegen_fn_params(args: &[Argument]) -> TokenStream {
    let params: Vec<_> = args
        .iter()
        .map(|arg| {
            let name = arg_ident(arg);
            let ty = arg_type_tokens(arg);
            quote! { #name: #ty }
        })
        .collect();

    quote! { #(#params),* }
}

/// Generate return type from output arguments
///
/// Single output: `Type`
/// Multiple outputs: `(Type1, Type2, ...)`
pub fn codegen_return_type(outputs: &[Argument]) -> TokenStream {
    if outputs.len() == 1 {
        arg_type_tokens(&outputs[0])
    } else {
        let types: Vec<_> = outputs.iter().map(arg_type_tokens).collect();
        quote! { (#(#types),*) }
    }
}

/// Generate return expression from output arguments
///
/// Single output: `name`
/// Multiple outputs: `(name1, name2, ...)`
pub fn codegen_return_expr(outputs: &[Argument]) -> TokenStream {
    if outputs.len() == 1 {
        let name = arg_ident(&outputs[0]);
        quote! { #name }
    } else {
        let names: Vec<_> = outputs.iter().map(arg_ident).collect();
        quote! { (#(#names),*) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_ir::ir::{BoolStore, TensorType};

    fn tensor_arg(name: &str, rank: usize, dtype: DType) -> Argument {
        Argument::new(name, ArgType::Tensor(TensorType::new(dtype, rank, None)))
    }

    #[test]
    fn args_doc_documents_tensor_dtype() {
        // A tensor's Rust signature carries the rank but not the element dtype, so the
        // doc line is the only place the ONNX-declared dtype is visible.
        let args = [
            tensor_arg("audio", 3, DType::F32),
            tensor_arg("ids", 2, DType::I64),
        ];

        let doc = codegen_args_doc(&args).unwrap().to_string();
        assert!(doc.contains("expected dtype"), "{doc}");
        assert!(doc.contains("`audio`: expected dtype `F32`"), "{doc}");
        assert!(doc.contains("`ids`: expected dtype `I64`"), "{doc}");
    }

    #[test]
    fn args_doc_names_bool_without_the_store_variant() {
        let args = [tensor_arg("mask", 2, DType::Bool(BoolStore::Native))];

        let doc = codegen_args_doc(&args).unwrap().to_string();
        // `Bool(Native)` is a storage layout, not something a caller passes.
        assert!(doc.contains("`mask`: expected dtype `bool`"), "{doc}");
        assert!(!doc.contains("Native"), "{doc}");
    }

    #[test]
    fn args_doc_is_omitted_when_there_are_no_inputs() {
        assert!(codegen_args_doc(&[]).is_none());
    }

    #[test]
    fn tensor_type_tokens_preserves_tensor_kind() {
        assert_eq!(
            tensor_type_tokens(3, &DType::F32).to_string(),
            quote!(Tensor<3>).to_string()
        );
        assert_eq!(
            tensor_type_tokens(3, &DType::I32).to_string(),
            quote!(Tensor<3, Int>).to_string()
        );
        assert_eq!(
            tensor_type_tokens(3, &DType::Bool(BoolStore::Native)).to_string(),
            quote!(Tensor<3, Bool>).to_string()
        );
    }

    #[test]
    fn scalar_type_tokens_float_types() {
        assert_eq!(
            scalar_type_tokens(&DType::F16).to_string(),
            quote!(half::f16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::BF16).to_string(),
            quote!(half::bf16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::F32).to_string(),
            quote!(f32).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::F64).to_string(),
            quote!(f64).to_string()
        );
    }

    #[test]
    fn scalar_type_tokens_signed_int_types() {
        assert_eq!(
            scalar_type_tokens(&DType::I8).to_string(),
            quote!(i8).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::I16).to_string(),
            quote!(i16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::I32).to_string(),
            quote!(i32).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::I64).to_string(),
            quote!(i64).to_string()
        );
    }

    #[test]
    fn scalar_type_tokens_unsigned_int_types() {
        assert_eq!(
            scalar_type_tokens(&DType::U8).to_string(),
            quote!(u8).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::U16).to_string(),
            quote!(u16).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::U32).to_string(),
            quote!(u32).to_string()
        );
        assert_eq!(
            scalar_type_tokens(&DType::U64).to_string(),
            quote!(u64).to_string()
        );
    }

    #[test]
    fn scalar_type_tokens_bool() {
        assert_eq!(
            scalar_type_tokens(&DType::Bool(BoolStore::Native)).to_string(),
            quote!(bool).to_string()
        );
    }
}
