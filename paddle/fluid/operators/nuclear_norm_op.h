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

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/linalg_ops_common.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/jacobi_svd.h"

namespace paddle {
namespace operators {

// U*S*V^T = JacobiSVD(X)
// out = sum(Z)
template <typename DeviceContext, typename T>
class NuclearNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    framework::Tensor *Out = context.Output<framework::Tensor>("Out");
    framework::Tensor *U = context.Output<framework::Tensor>("U");
    framework::Tensor *V = context.Output<framework::Tensor>("V");
    framework::Tensor S;

    Out->mutable_data<T>(context.GetPlace());
    math::JacobiSVDFunctor<DeviceContext, T> svd;
    auto &dev_ctx = context.template device_context<DeviceContext>();
    svd(dev_ctx, X, U, &S, V);

    VLOG(1) << "Tensor S:" << S;
    auto s = framework::EigenVector<T>::From(S);
    auto out = framework::EigenScalar<T>::From(*Out);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    out.device(place) = s.sum();
  }
};

// dX = dout * U * V^T
template <typename DeviceContext, typename T>
class NuclearNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const framework::Tensor *X = context.Input<framework::Tensor>("X");
    const framework::Tensor *U = context.Input<framework::Tensor>("U");
    const framework::Tensor *V = context.Input<framework::Tensor>("V");
    const framework::Tensor *d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor *dx =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    VLOG(1) << "d_out: " << *d_out;
    VLOG(1) << "U" << *U;
    VLOG(1) << "V" << *V;

    framework::Tensor Ret;
    Ret.mutable_data<T>(X->dims(), context.GetPlace());

    PADDLE_ENFORCE_EQ(d_out->numel(), 1UL,
                      platform::errors::InvalidArgument(
                          "Nuclear Norm Gradient should be scalar"));

    VLOG(1) << "NuclearNormGradKernel malloc for dx";
    dx->mutable_data<T>(context.GetPlace());
    VLOG(1) << "NuclearNormGradKernel malloc over";

    auto u_eigen = framework::EigenMatrix<T>::From(*U);
    auto v_eigen = framework::EigenMatrix<T>::From(*V);
    auto dx_eigen = framework::EigenMatrix<T>::From(*dx);
    auto ret_eigen = framework::EigenMatrix<T>::From(Ret);
    // d_out->Resize({1,1});
    auto d_out_eigen =
        framework::EigenMatrix<T>::From(*d_out, {d_out->numel(), 1});

    Eigen::array<Eigen::IndexPair<int>, 1> transposed_product_dims = {
        Eigen::IndexPair<int>(1, 1)};
    ret_eigen = u_eigen.contract(v_eigen, transposed_product_dims);

    VLOG(1) << "ret: " << ret_eigen;

    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    // Eigen::array<int, 2> ret_size({3, 2});
    auto ret_size = framework::EigenDim<2>::From(X->dims());
    dx_eigen.device(place) = d_out_eigen.broadcast(ret_size) * ret_eigen;

    /*
    auto x_eigen = framework::EigenVector<T>::Flatten(*x);
    auto d_out_eigen = framework::EigenVector<T>::Flatten(*d_out);
    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
    auto &place =
        *context.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 1> x_dsize(x->numel());
    dx_eigen.device(place) = d_out_eigen.broadcast(x_dsize) * x_eigen.sign();
    */
  }
};

}  // namespace operators
}  // namespace paddle
