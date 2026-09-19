#pragma once
#include "reconstruction.cuh"

// Recover complex Fourier coefficients from four phase-shifted frames.
__global__ void fourierCoefficients(const float *raw, float2 *coeff, int count, int nf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * nf) {
        const float *v = raw + 4 * i;
        coeff[i] = make_float2((v[0] - v[2]) * .5f, (v[3] - v[1]) * .5f);
    }
}
// Evaluate the sparse Fourier response at one projector position.
__device__ float response(const float2 *coeff, int p, int nf, float x, float y) {
    float re = 0, im = 0;
    for (int k = 0; k < nf; ++k)
        for (int s = 1; s >= -1; s -= 2) {
            float2 c = coeff[p * nf + k];
            c.y *= s;
            float pr = 2 * PI * (s * MF[k].y) * (y - .5f) / N;
            float pc = 2 * PI * (s * MF[k].x) * (x - .5f) / N;
            float rr = cosf(pr), ri = sinf(pr), cr = cosf(pc), ci = sinf(pc);
            float br = rr * cr - ri * ci, bi = rr * ci + ri * cr;
            re += c.x * br - c.y * bi;
            im += c.x * bi + c.y * br;
        }
    return sqrtf(re * re + im * im);
}
// Refine the selected peak with a nine-point quadratic surface fit.
__global__ void quadratic2D(const float2 *coeff, const float2 *coarse, float2 *match, int count,
                            int nf) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    float2 uv = coarse[p];
    match[p] = uv;
    if (!isfinite(uv.x) || uv.x < 1 || uv.y < 1 || uv.x > N - 2 || uv.y > N - 2)
        return;
    float z[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            z[i][j] = response(coeff, p, nf, uv.x + i - 1, uv.y + j - 1);
    float gx = (z[2][1] - z[0][1]) * .5f, gy = (z[1][2] - z[1][0]) * .5f;
    float hxx = z[2][1] - 2 * z[1][1] + z[0][1], hyy = z[1][2] - 2 * z[1][1] + z[1][0];
    float hxy = (z[2][2] - z[2][0] - z[0][2] + z[0][0]) * .25f;
    float det = hxx * hyy - hxy * hxy;
    if (hxx < -1e-6f && hyy < -1e-6f && det > 1e-6f) {
        float du = (hxy * gy - hyy * gx) / det, dv = (hxy * gx - hxx * gy) / det;
        match[p] =
            make_float2(uv.x + fminf(.5f, fmaxf(-.5f, du)), uv.y + fminf(.5f, fmaxf(-.5f, dv)));
    }
}
// Search integer-centered neighborhoods along the finite near/far segment.
__global__ void centerSearch(const float2 *coeff, const Geometry *geometry, float2 *coarse,
                             int *counts, int count, int nf, int centers, int halfWidth,
                             float peakRatio) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    Geometry g = geometry[p];
    float dx = g.bx - g.ax, dy = g.by - g.ay;
    float length2 = dx * dx + dy * dy;
    bool horizontal = fabsf(dx) >= fabsf(dy);
    float amin = horizontal ? fminf(g.ax, g.bx) : fminf(g.ay, g.by);
    float amax = horizontal ? fmaxf(g.ax, g.bx) : fmaxf(g.ay, g.by);
    int lo = max(0, int(ceilf(amin))), hi = min(N - 1, int(floorf(amax)));
    int nc = min(centers, max(0, hi - lo + 1));
    float best = -INFINITY;
    float2 uv = make_float2(NAN, NAN);
    int n = 0;
    for (int ci = 0; ci < nc; ++ci) {
        int major = lo + (nc > 1 ? int(roundf(float(ci) * (hi - lo) / (nc - 1))) : 0);
        int minor =
            int(roundf(horizontal ? -(g.la * major + g.lc) / g.lb : -(g.lb * major + g.lc) / g.la));
        int cx = horizontal ? major : minor, cy = horizontal ? minor : major;
        for (int offset = 0; offset < (2 * halfWidth + 1) * (2 * halfWidth + 1); ++offset) {
            int x = cx + offset % (2 * halfWidth + 1) - halfWidth,
                y = cy + offset / (2 * halfWidth + 1) - halfWidth;
            if (x < 0 || x >= N || y < 0 || y >= N)
                continue;
            float along = ((x - g.ax) * dx + (y - g.ay) * dy) / length2;
            if (along < 0 || along > 1)
                continue;
            ++n;
            float value = response(coeff, p, nf, float(x), float(y));
            if (value > best) {
                best = value;
                uv = make_float2(float(x), float(y));
            }
        }
    }
    // Keep strong local maxima and resolve periodic ambiguity using epipolar distance.
    float bestDistance = INFINITY, selectedValue = -INFINITY;
    float2 selected = make_float2(NAN, NAN);
    for (int ci = 0; ci < nc; ++ci) {
        int major = lo + (nc > 1 ? int(roundf(float(ci) * (hi - lo) / (nc - 1))) : 0);
        int minor =
            int(roundf(horizontal ? -(g.la * major + g.lc) / g.lb : -(g.lb * major + g.lc) / g.la));
        int cx = horizontal ? major : minor, cy = horizontal ? minor : major;
        for (int offset = 0; offset < (2 * halfWidth + 1) * (2 * halfWidth + 1); ++offset) {
            int x = cx + offset % (2 * halfWidth + 1) - halfWidth,
                y = cy + offset / (2 * halfWidth + 1) - halfWidth;
            if (x < 0 || x >= N || y < 0 || y >= N)
                continue;
            float along = ((x - g.ax) * dx + (y - g.ay) * dy) / length2;
            if (along < 0 || along > 1)
                continue;

            float value = response(coeff, p, nf, float(x), float(y));
            if (value < peakRatio * best || x < 1 || y < 1 || x > N - 2 || y > N - 2)
                continue;
            bool peak = true;
            for (int oy = -1; oy <= 1 && peak; ++oy)
                for (int ox = -1; ox <= 1; ++ox) {
                    if (!ox && !oy)
                        continue;
                    float neighbor = response(coeff, p, nf, float(x + ox), float(y + oy));
                    if (neighbor > value) {
                        peak = false;
                        break;
                    }
                }
            if (!peak)
                continue;
            // Fit the nine neighboring samples before comparing epipolar distances.
            float z[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    z[i][j] = response(coeff, p, nf, float(x + i - 1), float(y + j - 1));
            float gx = (z[2][1] - z[0][1]) * .5f, gy = (z[1][2] - z[1][0]) * .5f;
            float hxx = z[2][1] - 2 * z[1][1] + z[0][1], hyy = z[1][2] - 2 * z[1][1] + z[1][0];
            float hxy = (z[2][2] - z[2][0] - z[0][2] + z[0][0]) * .25f;
            float det = hxx * hyy - hxy * hxy;
            if (!(hxx < -1e-6f && hyy < -1e-6f && det > 1e-6f))
                continue;
            float du = fminf(.5f, fmaxf(-.5f, (hxy * gy - hyy * gx) / det));
            float dv = fminf(.5f, fmaxf(-.5f, (hxy * gx - hxx * gy) / det));
            float distance = fabsf(g.la * (x + du) + g.lb * (y + dv) + g.lc);
            if (distance < bestDistance || (distance == bestDistance && value > selectedValue)) {
                bestDistance = distance;
                selectedValue = value;
                selected = make_float2(float(x), float(y));
            }
        }
    }
    uv = selected;
    coarse[p] = uv;
    counts[p] = n;
}
