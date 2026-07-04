#ifndef COMMON_H
#define COMMON_H

#include <metal_stdlib>
using namespace metal;

struct LionParams {
  float lr;
  float beta1;
  float beta2;
  float weight_decay;
};

#endif  // COMMON_H