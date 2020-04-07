/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/jacobi_svd.h"

namespace paddle {
namespace operators {
namespace math {

// U*Z*V^T = JacobiSVD(X)
template <typename DeviceContext, typename T>
void JacobiSVDFunctor<DeviceContext, T>::operator()(
    const DeviceContext& context, const framework::Tensor* X,
    framework::Tensor* U, framework::Tensor* S, framework::Tensor* V) {
  VLOG(1) << "calling JacobiSVDFunctor";
  PADDLE_ENFORCE_EQ(X->dims().size(), 2UL,
                    platform::errors::InvalidArgument(
                        "The input tensor X's "
                        "dimensions of JacobiSVDFunctor should be 2UL, "
                        "But received X's dimensions = %d, X's shape = [%s]",
                        X->dims().size(), X->dims()));

  auto xdim = X->dims();
  auto min_dim = xdim[0] < xdim[1] ? xdim[0] : xdim[1];
  auto udim = framework::make_ddim({xdim[0], min_dim});
  auto sdim = framework::make_ddim({min_dim});
  auto vdim = framework::make_ddim({xdim[1], min_dim});

  U->mutable_data<T>(udim, context.GetPlace());
  S->mutable_data<T>(sdim, context.GetPlace());
  V->mutable_data<T>(vdim, context.GetPlace());

  auto x = framework::MatrixEigen<T>::From(*X);

  auto u = framework::MatrixEigen<T>::From(*U, udim);
  auto s = framework::MatrixEigen<T>::From(*S, sdim);
  auto v = framework::MatrixEigen<T>::From(*V, vdim);

  // auto &place = context.eigen_device();

  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  VLOG(1) << "calling svd";
  Eigen::JacobiSVD<Matrix> svd(x, Eigen::ComputeThinU | Eigen::ComputeThinV);
  VLOG(1) << "end svd call";
  // framework::MatrixEigen<T>::From(*U, udim) = svd.matrixU();
  u = svd.matrixU();
  VLOG(1) << "U end";
  // framework::VectorEigen<T>::From(*S, sdim) = svd.singularValues();
  s = svd.singularValues();
  VLOG(1) << "S end";
  // framework::MatrixEigen<T>::From(*V, vdim) = svd.matrixV();
  v = svd.matrixV();
  VLOG(1) << "V end";
}

template class JacobiSVDFunctor<platform::CPUDeviceContext, float>;
template class JacobiSVDFunctor<platform::CPUDeviceContext, double>;
template class JacobiSVDFunctor<platform::CUDADeviceContext, float>;
template class JacobiSVDFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
