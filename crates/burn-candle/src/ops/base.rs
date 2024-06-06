use core::iter;
use std::{borrow::Borrow, marker::PhantomData, sync::Arc};

use burn_tensor::{backend::Backend, Data, Reader, Shape};
use candle_core::IndexOp;

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleDevice, CandleTensor,
};

use super::tensor;

pub fn cat<E: CandleElement, const D: usize>(
    tensors: Vec<CandleTensor<E, D>>,
    dim: usize,
) -> CandleTensor<E, D> {
    let tensors: Vec<candle_core::Tensor> = tensors.into_iter().map(|t| t.tensor).collect();
    CandleTensor::new(candle_core::Tensor::cat(&tensors, dim).unwrap())
}

pub fn from_data<E: CandleElement, const D: usize>(
    data: Data<E, D>,
    device: &CandleDevice,
) -> CandleTensor<E, D> {
    CandleTensor::from_data(data, *device)
}
pub fn into_data<E: CandleElement, const D: usize>(tensor: CandleTensor<E, D>) -> Data<E, D> {
    Data::new(
        tensor.tensor.flatten_all().unwrap().to_vec1().unwrap(),
        tensor.shape(),
    )
}

pub fn diagonal<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    offset: i64,
    dim1: usize,
    dim2: usize,
) -> CandleTensor<E, D2> {
    // FIXME: Replace with an appropriate method when Candle provides one.
    let tensor = tensor.tensor;
    let shape = tensor.dims();

    fn get_dim_size(tensor: &candle_core::Tensor, dim: usize) -> usize {
        tensor
            .dim(dim)
            .unwrap_or_else(|err| panic!("Requested dimension '{:?}' not found: {}", dim, err))
    }

    let dim1_size = get_dim_size(&tensor, dim1);
    let dim2_size = get_dim_size(&tensor, dim2);

    let diag_size = match offset >= 0 {
        true => usize::min(
            dim1_size,
            dim2_size.saturating_sub(offset.unsigned_abs() as usize),
        ),
        false => usize::min(
            dim1_size.saturating_sub(offset.unsigned_abs() as usize),
            dim2_size,
        ),
    };

    // 1. Permute dim1 and dim2 to the start
    let perm_indices = core::iter::once(dim1)
        .chain(core::iter::once(dim2))
        .chain((0..shape.len()).filter(|&i| i != dim1 && i != dim2))
        .collect::<Vec<usize>>();

    let tensor = tensor.permute(perm_indices.as_slice()).unwrap();

    // 1.5. Get shape of output tensor
    let shape = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != dim1 && i != dim2)
        .map(|(_, &s)| s)
        .chain(iter::once(diag_size))
        .collect::<Vec<_>>();

    // 2. Get diagonal indices as ranges
    let i = (0..diag_size);
    let j = (offset.unsigned_abs() as usize..diag_size + offset.unsigned_abs() as usize);

    // 3. Get diagonals
    let mut diags = Vec::new();
    for (i, j) in i.into_iter().zip(j.into_iter()) {
        // This is why we permuted
        let index = match offset >= 0 {
            true => (i, j),
            false => (j, i),
        };
        let diag = tensor
            .i(index)
            .expect("Candle index error")
            .flatten_all()
            .unwrap()
            .reshape(((), 1))
            .unwrap();

        diags.push(diag);
    }

    // Concat and reshape resulting tensor vectors
    let new_tensor = candle_core::Tensor::cat(diags.as_slice(), 1)
        .unwrap()
        .reshape(shape)
        .unwrap()
        .to_owned();

    CandleTensor::new(new_tensor)
}

pub fn to_device<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    device: &CandleDevice,
) -> CandleTensor<E, D> {
    CandleTensor::new(tensor.tensor.to_device(&(*device).into()).unwrap())
}

pub fn empty<E: CandleElement, const D: usize>(
    shape: Shape<D>,
    device: &CandleDevice,
) -> CandleTensor<E, D> {
    CandleTensor::new(candle_core::Tensor::zeros(&shape.dims, E::DTYPE, &(*device).into()).unwrap())
}

pub fn swap_dims<E: CandleElement, const D: usize>(
    mut tensor: CandleTensor<E, D>,
    dim1: usize,
    dim2: usize,
) -> CandleTensor<E, D> {
    CandleTensor::new(tensor.tensor.transpose(dim1, dim2).unwrap())
}

pub fn permute<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    axes: [usize; D],
) -> CandleTensor<E, D> {
    CandleTensor::new(tensor.tensor.permute(axes).unwrap())
}

pub fn flip<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    axes: &[usize],
) -> CandleTensor<E, D> {
    // FIXME: Replace with an appropriate method when Candle provides one.
    let mut tensor = tensor.tensor;
    for &axis in axes {
        let indexes = candle_core::Tensor::arange_step(
            tensor.dim(axis).unwrap() as i64 - 1,
            -1,
            -1,
            tensor.device(),
        )
        .unwrap();
        tensor = tensor.index_select(&indexes, axis).unwrap();
    }

    CandleTensor::new(tensor)
}

pub fn reshape<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    shape: Shape<D2>,
) -> CandleTensor<E, D2> {
    CandleTensor::new(tensor.tensor.reshape(&shape.dims).unwrap())
}

pub fn device<E: CandleElement, const D: usize>(tensor: &CandleTensor<E, D>) -> CandleDevice {
    tensor.tensor.device().clone().into()
}

pub fn shape<E: CandleElement, const D: usize>(tensor: &CandleTensor<E, D>) -> Shape<D> {
    tensor.shape()
}

pub fn slice<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    ranges: [std::ops::Range<usize>; D2],
) -> CandleTensor<E, D1> {
    let mut narrow_tensor = tensor.tensor;
    for (i, range) in ranges.iter().enumerate().take(D2) {
        narrow_tensor = narrow_tensor
            .narrow(i, range.start, range.end - range.start)
            .unwrap()
    }
    CandleTensor::new(narrow_tensor)
}

pub fn slice_assign<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    ranges: [std::ops::Range<usize>; D2],
    value: CandleTensor<E, D1>,
) -> CandleTensor<E, D1> {
    CandleTensor::new(tensor.tensor.slice_assign(&ranges, &value.tensor).unwrap())
}

pub fn narrow<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    dim: usize,
    start: usize,
    length: usize,
) -> CandleTensor<E, D> {
    let tensor = tensor.tensor.narrow(dim, start, length);
    match tensor {
        Ok(tensor) => CandleTensor::new(tensor),
        Err(e) => panic!("error narrow from Candle"),
    }
}

pub fn chunk<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    chunks: usize,
    dim: usize,
) -> Vec<CandleTensor<E, D>> {
    let tensors = tensor.tensor.chunk(chunks, dim);
    match tensors {
        Ok(tensors) => tensors
            .into_iter()
            .map(|tensor| CandleTensor::new(tensor))
            .collect(),
        Err(e) => panic!("error chunk from Candle"),
    }
}

pub fn expand<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    shape: Shape<D2>,
) -> CandleTensor<E, D2> {
    CandleTensor::new(tensor.tensor.broadcast_as(&shape.dims).unwrap())
}

pub fn sign<E: CandleElement, const D: usize>(tensor: CandleTensor<E, D>) -> CandleTensor<E, D> {
    CandleTensor::new(tensor.tensor.sign().unwrap())
}
