//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

float MinusOne(float value) { return value - 1.f; }
TEST(test_transform, transform1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Rand();
  f1.Show();
  f1.Transform(MinusOne);
  f1.Show();
}

float ReLu(float value) { return std::max(0.f, value); }
TEST(test_transform, transform2) {
  using namespace kuiper_infer;
  Tensor<float> f1(3, 4, 5);
  f1.Rand();
  f1.Show();
  f1.Transform(ReLu);
  f1.Show();
}