/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <Eigen/Core>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace framework {

// Interpret paddle::platform::Tensor as MatrixEigen.
template <typename T, int MajorType = Eigen::RowMajor>
struct MatrixEigen {
  using Type =
      Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, MajorType>>;

  using ConstType = Eigen::Map<
      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, MajorType>>;

  static Type From(Tensor& tensor, DDim dims) {  // NOLINT
    PADDLE_ENFORCE_EQ(
        dims.size(), 2UL,
        platform::errors::InvalidArgument(
            "The input tensor's "
            "dimensions of MatrixEigen should be 2UL, "
            "But received Tensor's dimensions = %d, Tensor's shape = [%s]",
            dims.size(), dims));
    // PADDLE_ENFORCE(dims.size() == 2, "Demension of MatrixEigen must equal
    // 2");
    return Type(tensor.data<T>(), dims[0], dims[1]);
  }

  static Type From(Tensor& tensor) {  // NOLINT
    return From(tensor, tensor.dims());
  }  // NOLINT

  static ConstType From(const Tensor& tensor, DDim dims) {
    PADDLE_ENFORCE_EQ(
        dims.size(), 2UL,
        platform::errors::InvalidArgument(
            "The input tensor's "
            "dimensions of MatrixEigen should be 2UL, "
            "But received Tensor's dimensions = %d, Tensor's shape = [%s]",
            dims.size(), dims));
    // PADDLE_ENFORCE(dims.size() == 2, "Demension of MatrixEigen must equal
    // 2");
    return ConstType(tensor.data<T>(), dims[0], dims[1]);
  }

  static ConstType From(const Tensor& tensor) {
    return From(tensor, tensor.dims());
  }
};

// Interpret paddle::platform::Tensor as VectorEigen.
template <typename T, int MajorType = Eigen::RowMajor>
struct VectorEigen {
  using Type = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, MajorType>>;

  using ConstType =
      Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, MajorType>>;

  static Type From(Tensor& tensor, DDim dims) {  // NOLINT
    PADDLE_ENFORCE_EQ(
        dims.size(), 1UL,
        platform::errors::InvalidArgument(
            "The input tensor's "
            "dimensions of VectorEigen should be 1UL, "
            "But received Tensor's dimensions = %d, Tensor's shape = [%s]",
            dims.size(), dims));
    // PADDLE_ENFORCE(dims.size() == 2, "Demension of MatrixEigen must equal
    // 2");
    return Type(tensor.data<T>(), dims[0]);
  }

  static Type From(Tensor& tensor) {  // NOLINT
    return From(tensor, tensor.dims());
  }  // NOLINT

  static ConstType From(const Tensor& tensor, DDim dims) {
    PADDLE_ENFORCE_EQ(
        dims.size(), 1UL,
        platform::errors::InvalidArgument(
            "The input tensor's "
            "dimensions of VectorEigen should be 1UL, "
            "But received Tensor's dimensions = %d, Tensor's shape = [%s]",
            dims.size(), dims));
    // PADDLE_ENFORCE(dims.size() == 2, "Demension of MatrixEigen must equal
    // 2");
    return ConstType(tensor.data<T>(), dims[0]);
  }

  static ConstType From(const Tensor& tensor) {
    return From(tensor, tensor.dims());
  }
};

}  // namespace framework
}  // namespace paddle
