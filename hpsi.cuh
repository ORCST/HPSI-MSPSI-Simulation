#pragma once
#include "reconstruction.cuh"

// Subtract the DC frame to obtain the Hartley coefficients.
__global__ void hartleyCoefficients(const float *raw, float *coeff, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * 20)
        coeff[i] = raw[(i / 20) * 21 + i % 20] - raw[(i / 20) * 21 + 20];
}


// Restore the EN and UN profiles on the 512-sample basis grid.
__global__ void hProfiles512(const float *coeff, const float *hbasis, float *h, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * 512)
        return;
    int p = i / 512, x = i % 512;
    for (int axis = 0; axis < 2; ++axis) {
        float en = 0, un = 0;
        for (int k = 0; k < 10; ++k) {
            float z = coeff[p * 20 + axis * 10 + k] * hbasis[k * 512 + x];
            en += z;
            if (HF[k / 2] % 16 == 0)
                un += z;
        }
        size_t o = size_t(p) * 4 * 512 + x;
        h[o + axis * 512] = en;
        h[o + (axis + 2) * 512] = un;
    }
}
__device__ bool hLocalPeak(const float *v, int x) {
    return x > 0 && x < 512 - 1 && v[x] >= v[x - 1] && v[x] > v[x + 1];
}
// Select local EN peaks within the confidence interval and depth bounds.
__global__ void hSearch512(const float *h, const Geometry *g, float2 *coarse, int *candidates,
                           int count, int radius) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    const float *u = h + size_t(p) * 4 * 512, *v = u + 512;
    int ux = 0, vy = 0;
    for (int x = 1; x < 512; ++x) {
        if (u[x] > u[ux])
            ux = x;
        if (v[x] > v[vy])
            vy = x;
    }
    int us[129], vs[129], nu = 0, nv = 0;
    for (int x = max(1, ux - radius); x <= min(512 - 2, ux + radius); ++x)
        if (hLocalPeak(u, x))
            us[nu++] = x;
    for (int y = max(1, vy - radius); y <= min(512 - 2, vy + radius); ++y)
        if (hLocalPeak(v, y))
            vs[nv++] = y;
    // Select correspondences based on epipolar distance.
    float best = INFINITY;
    float2 uv = make_float2(NAN, NAN);
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j) {
            float dx = g[p].bx - g[p].ax, dy = g[p].by - g[p].ay;
            float along = ((us[i] - g[p].ax) * dx + (vs[j] - g[p].ay) * dy) / (dx * dx + dy * dy);
            if (along < 0 || along > 1)
                continue;
            float d = fabsf(g[p].la * us[i] + g[p].lb * vs[j] + g[p].lc);
            if (d < best) {
                best = d;
                uv = make_float2(float(us[i]), float(vs[j]));
            }
        }
    candidates[p] = nu * nv;
    coarse[p] = uv;
}
__device__ float hParabola(float l, float c, float r) {
    float den = l - 2 * c + r;
    return den < -1e-6f ? fminf(.5f, fmaxf(-.5f, .5f * (l - r) / den)) : 0;
}
// Refine the UN peak and convert back to physical projector coordinates.
__global__ void hRefine512(const float *h, const float2 *coarse, float2 *match, int count,
                           int radius) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    if (!isfinite(coarse[p].x)) {
        match[p] = coarse[p];
        return;
    }
    float result[2];
    for (int a = 0; a < 2; ++a) {
        const float *v = h + size_t(p) * 4 * 512 + (a + 2) * 512;
        int center = int(a ? coarse[p].y : coarse[p].x);
        int lo = max(1, center - radius), hi = min(512 - 2, center + radius), best = lo;
        for (int x = lo + 1; x <= hi; ++x)
            if (v[x] > v[best])
                best = x;
        result[a] = best + hParabola(v[best - 1], v[best], v[best + 1]);
    }
    match[p] = make_float2(2 * result[0] + .5f, 2 * result[1] + .5f);
}
