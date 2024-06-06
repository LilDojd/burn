#[burn_tensor_testgen::testgen(diagonal)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Data, Int, Tensor};

    #[test]
    fn should_support_diagonal_2d_main() {
        let data = Data::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());

        let diagonal = tensor.diagonal::<1>(0, 0, 1);
        let data_expected = Data::from([0, 4, 8]);
        assert_eq!(data_expected, diagonal.into_data());
    }

    #[test]
    fn should_support_nonsquare_tensors() {
        let data = Data::from([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());

        let diagonal = tensor.clone().diagonal::<1>(0, 0, 1);
        let data_expected = Data::from([0, 4, 8]);
        assert_eq!(data_expected, diagonal.into_data());

        // -1 offset
        let diagonal = tensor.clone().diagonal::<1>(-1, 0, 1);
        let data_expected = Data::from([3, 7, 11]);
        assert_eq!(data_expected, diagonal.into_data());

        // +1 offset
        let diagonal = tensor.diagonal::<1>(1, 0, 1);
        let data_expected = Data::from([1, 5]);
        assert_eq!(data_expected, diagonal.into_data());
    }

    #[test]
    fn should_support_diagonal_2d_offset_positive() {
        let data = Data::from([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let diagonal = tensor.diagonal::<1>(1, 0, 1);
        let data_expected = Data::from([1., 5.]);
        assert_eq!(data_expected, diagonal.into_data());
    }

    #[test]
    fn should_support_diagonal_2d_offset_negative() {
        let data = Data::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());

        let diagonal = tensor.diagonal::<1>(-1, 0, 1);
        let data_expected = Data::from([3, 7]);
        assert_eq!(data_expected, diagonal.into_data());
    }

    #[test]
    fn should_support_outermost_diagonal() {
        let data = Data::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());

        let diagonal = tensor.diagonal::<1>(2, 0, 1);
        let data_expected = Data::from([2]);
        assert_eq!(data_expected, diagonal.into_data());
    }

    #[test]
    fn should_support_diagonal_3d() {
        let data = Data::from([
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
        ]);
        let tensor = Tensor::<TestBackend, 3, Int>::from_data(data, &Default::default());

        let diagonal = tensor.diagonal::<2>(0, 1, 2);
        let data_expected = Data::from([[0, 4, 8], [9, 13, 17], [18, 22, 26]]);
        assert_eq!(data_expected, diagonal.into_data());
    }

    #[test]
    fn should_return_empty_on_large_offset() {
        let data = Data::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());

        let diagonal = tensor.diagonal::<1>(10, 0, 1);
        assert_eq!(diagonal.dims().len(), 1);
        assert_eq!(diagonal.dims()[0], 0);
    }

    #[test]
    fn should_return_empty_on_large_offset_6d() {
        let tensor: Tensor<TestBackend, 6> = Tensor::ones([3, 3, 3, 3, 3, 3], &Default::default());
        let diagonal = tensor.diagonal::<5>(-10, 0, 1);
        assert_eq!(diagonal.dims().len(), 5);
        assert_eq!(diagonal.dims().last().unwrap(), &0);
    }

    #[test]
    fn should_support_bool() {
        let data = Data::from([
            [true, false, true],
            [false, true, false],
            [true, false, true],
        ]);
        let tensor = Tensor::<TestBackend, 2, Bool>::from_data(data, &Default::default());

        let diagonal = tensor.diagonal::<1>(0, 0, 1);
        let data_expected = Data::from([true, true, true]);
        assert_eq!(data_expected, diagonal.into_data());
    }

    #[test]
    #[should_panic]
    fn one_d_diagonal() {
        let data = Data::from([0.0, 1.0, 2.0]);
        let tensor = Tensor::<TestBackend, 1>::from_data(data, &Default::default());
        let _ = tensor.diagonal::<0>(0, 0, 1);
    }

    #[test]
    #[should_panic]
    fn same_dims() {
        let data = Data::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());
        let _ = tensor.diagonal::<1>(0, 0, 0);
    }

    #[test]
    #[should_panic]
    fn dims_out_of_bounds() {
        let data = Data::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let tensor = Tensor::<TestBackend, 2, Int>::from_data(data, &Default::default());
        let _ = tensor.diagonal::<1>(0, 0, 3);
    }
}
