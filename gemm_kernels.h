#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>

constexpr size_t BLOCK_SIZE = 256;

// Only square matrices are supported
void launchEx1A(float* C, const float* A, const float* B, size_t size);
void launchEx1B(float* C, const float* A, const float* B, size_t size);
void launchEx2(float* c, const float* A, const float* b, size_t size);
