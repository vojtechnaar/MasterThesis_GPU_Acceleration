#pragma once

#include <cstdio>
#include <cstdlib>
#include <chrono>

/* ─── CUDA error handling ─── */
#define CUDA_OK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d — %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* ─── timing helpers ─── */
using Clock = std::chrono::high_resolution_clock;

static inline double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/* ─── algorithm constants ─── */
static constexpr int MAXVAR   = 16;   /* max distinct variables per rule      */
static constexpr int MAX_BODY = 8;    /* max body atoms per rule              */
static constexpr int COOP_FAN = 64;   /* fan-out threshold for warp-coop kernel */
