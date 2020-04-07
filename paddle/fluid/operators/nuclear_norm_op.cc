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

#include "paddle/fluid/operators/nuclear_norm_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

// U*Z*V^T = JacobiSVD(X)
class NuclearNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("U"), "Output(U) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("V"), "Output(V) should be not null.");

    bool keep_dim = ctx->Attrs().Get<bool>("keep_dim");
    if (keep_dim) {
      ctx->SetOutputDim("Out", {1, 1});
    } else {
      ctx->SetOutputDim("Out", {1});
    }
  }
};

class NuclearNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of nuclear_norm op.");
    AddOutput("Out", "(Scalar) The output of nuclear_norm op.");
    AddOutput("U", "(Tensor) The  output of nuclear_norm op.").AsIntermediate();
    AddOutput("V", "(Tensor) The output of nuclear_norm op.").AsIntermediate();
    AddAttr<bool>(
        "keep_dim",
        "(bool, default false) "
        "Whether to reserve the reduced dimension in the output Tensor.")
        .SetDefault(false);
    AddComment(R"DOC(
Nuclear Norm Operator.

Computes the nuclear norm of a tensor.
$$U * \diag{\sigma} * V^T = X$$
$$Out = \sum{\sigma}$$

)DOC");
  }
};

class NuclearNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("U"), "Input(U) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("V"), "Input(V) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

template <typename T>
class NuclearNormOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("nuclear_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("U", this->Output("U"));
    op->SetInput("V", this->Output("V"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(nuclear_norm, ops::NuclearNormOp, ops::NuclearNormOpMaker,
                  ops::NuclearNormOpGradMaker<paddle::framework::OpDesc>,
                  ops::NuclearNormOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(nuclear_norm_grad, ops::NuclearNormGradOp);

REGISTER_OP_CPU_KERNEL(
    nuclear_norm,
    ops::NuclearNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NuclearNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    nuclear_norm_grad,
    ops::NuclearNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::NuclearNormGradKernel<paddle::platform::CPUDeviceContext, double>);
