#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        auto status_ = (call);                                                                     \
        if (status_ != cudaSuccess)                                                                \
            throw std::runtime_error(cudaGetErrorString(status_));                                 \
    } while (0)
template <class T> struct Device {
    T *p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device &) = delete;
    void upload(const T *host) {
        CUDA_CHECK(cudaMemcpy(p, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
};

constexpr int N = 1024, MAXF = 40;
constexpr float PI = 3.14159265358979323846f;
__constant__ int2 MF[MAXF];
__constant__ int HF[5];
__constant__ float INVKP[9], TRANS[3];
struct Geometry {
    float cu, cv, ax, ay, bx, by, la, lb, lc, rx, ry, rz, qx, qy, qz;
};
static_assert(sizeof(Geometry) == 15 * sizeof(float), "Geometry packing mismatch");
struct Truth {
    float u, v, x, y, z;
};
struct Times {
    double upload = 0, coeff = 0, profiles = 0, search = 0, subpixel = 0, triangulation = 0,
           download = 0, wall = 0;
};
template <class T> void read(std::ifstream &f, T *p, size_t n) {
    f.read(reinterpret_cast<char *>(p), n * sizeof(T));
    if (!f)
        throw std::runtime_error("Truncated input");
}
// Triangulate the camera and projector rays using the shared calibration.
__global__ void triangulate(const float2 *match, const Geometry *geometry, float3 *xyz, int count) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    Geometry g = geometry[p];
    float u = match[p].x + 449, v = match[p].y + 29;
    float q[3] = {g.qx, g.qy, g.qz}, rp[3];
    for (int k = 0; k < 3; ++k)
        rp[k] = INVKP[k * 3] * u + INVKP[k * 3 + 1] * v + INVKP[k * 3 + 2];
    float aa = 0, ab = 0, bb = 0, at = 0, bt = 0;
    for (int k = 0; k < 3; ++k) {
        aa += q[k] * q[k];
        ab -= q[k] * rp[k];
        bb += rp[k] * rp[k];
        at -= q[k] * TRANS[k];
        bt += rp[k] * TRANS[k];
    }
    float det = aa * bb - ab * ab, depth = (bb * at - ab * bt) / det,
          other = (aa * bt - ab * at) / det;
    xyz[p] = (isfinite(depth) && fabsf(det) > 1e-10f && depth > 0 && other > 0)
                 ? make_float3(g.rx * depth, g.ry * depth, g.rz * depth)
                 : make_float3(NAN, NAN, NAN);
}
// Deferred stage events avoid a host synchronization after every batch.
// Event resources are reused; result downloads provide common completion.
struct Timer {
    struct Slot {
        cudaEvent_t a, b;
        double *destination = nullptr;
    };
    std::vector<Slot> slots;
    size_t used = 0;
    ~Timer() {
        for (auto &s : slots) {
            cudaEventDestroy(s.a);
            cudaEventDestroy(s.b);
        }
    }
    void start() {
        if (used == slots.size()) {
            Slot s{};
            CUDA_CHECK(cudaEventCreate(&s.a));
            CUDA_CHECK(cudaEventCreate(&s.b));
            slots.push_back(s);
        }
        CUDA_CHECK(cudaEventRecord(slots[used].a));
    }
    void stop(double &destination) {
        auto &s = slots[used++];
        CUDA_CHECK(cudaEventRecord(s.b));
        s.destination = &destination;
        CUDA_CHECK(cudaGetLastError());
    }
    void resolve() {
        if (used)
            CUDA_CHECK(cudaEventSynchronize(slots[used - 1].b));
        for (size_t i = 0; i < used; ++i) {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, slots[i].a, slots[i].b));
            *slots[i].destination += ms;
        }
        used = 0;
    }
};

