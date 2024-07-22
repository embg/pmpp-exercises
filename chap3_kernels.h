#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>

constexpr size_t BLOCK_SIZE = 256;

void launchEx1A(float* C, const float* A, const float* B, size_t Width);
void launchEx1B(float* C, const float* A, const float* B, size_t Width);
