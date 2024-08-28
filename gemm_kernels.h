#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>

// For chap3 ex1A, ex1B
constexpr size_t BLOCK_SIZE = 256;

// For chap6
constexpr size_t TILE_WIDTH = 32;
constexpr size_t COARSE_FACTOR = 4;

// Only square matrices are supported
void launchEx1A(float* C, const float* A, const float* B, size_t size);
void launchEx1B(float* C, const float* A, const float* B, size_t size);
void launchEx2(float* c, const float* A, const float* b, size_t size);
void launchChap6(float* C, const float* A, const float* B, size_t stride);
