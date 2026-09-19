#pragma once

// Generate camera intensities from the moving surface and calibrated projector.
__constant__ float KP[9];
__global__ void simulate(const Geometry *g, const Truth *base, Truth *truth, float *rh, float *rm,
                         int n, float phase, float noise) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n)
        return;
    Geometry a = g[p];
    float z = base[p].z + 2 * sinf(phase) * cosf(a.cu * .012f);
    float q[3] = {a.qx * z + TRANS[0], a.qy * z + TRANS[1], a.qz * z + TRANS[2]}, pr[3];
    for (int j = 0; j < 3; j++)
        pr[j] = KP[j * 3] * q[0] + KP[j * 3 + 1] * q[1] + KP[j * 3 + 2] * q[2];
    float u = pr[0] / pr[2] - 449, v = pr[1] / pr[2] - 29;
    truth[p] = {u, v, a.rx * z, a.ry * z, a.rz * z};
    float amp = 32 + 8 * cosf(a.cu * .013f) * sinf(a.cv * .017f);
    for (int j = 0; j < 21; j++) {
        float value = 128;
        if (j < 20) {
            int aidx = j / 10, k = (j % 10) / 2;
            float f = HF[k] * (j % 2 ? -1 : 1);
            float w = 2 * PI * f / N;
            float ph = w * ((aidx ? v : u) - .5f);
            value += amp * expf(-.5f * w * w * 1.4f * 1.4f) * (cosf(ph) + sinf(ph));
        }
        unsigned r = (p * 1664525u + j * 1013904223u + 12345u);
        r ^= r >> 16;
        r *= 2246822519u;
        r ^= r >> 13;
        rh[p * 21 + j] = value + noise * ((r & 65535) / 32767.5f - 1) * 1.7320508f;
    }
    for (int k = 0; k < 10; k++) {
        float w = 2 * PI * (MF[k].x + MF[k].y) / N,
              ph = 2 * PI * (MF[k].x * (u - .5f) + MF[k].y * (v - .5f)) / N;
        float amp1 = amp * expf(-.5f * w * w * 1.4f * 1.4f), re = amp1 * cosf(ph),
              im = -amp1 * sinf(ph);
        for (int j = 0; j < 4; j++) {
            unsigned r = p * 1664525u + (4 * k + j) * 1013904223u + 98765u;
            r ^= r >> 16;
            r *= 2246822519u;
            r ^= r >> 13;
            rm[p * 40 + 4 * k + j] = 128 +
                                     (j == 0   ? re
                                      : j == 1 ? -im
                                      : j == 2 ? -re
                                               : im) +
                                     noise * ((r & 65535) / 32767.5f - 1) * 1.7320508f;
        }
    }
}
