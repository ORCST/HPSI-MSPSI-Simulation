#define NOMINMAX
#include "reconstruction.cuh"
#include "hpsi.cuh"
#include "mspsi.cuh"
#include "simulation.cuh"
#include <windows.h>
#include <GL/gl.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <sstream>
#include <filesystem>
#include "viewer.h"

struct Options {
    std::string input = "scene.bin";
    int frames = 0, centers = 300, half = 5;
    float noise = 0.4f;
    bool headless = false, animate = true;
};
void compute(Options o) {
    try {
        auto init = std::chrono::steady_clock::now();
        // Load the common ROI, calibration and reference surface.
        std::ifstream in(o.input, std::ios::binary);
        char magic[8];
        read(in, magic, 8);
        if (std::string(magic, 8) != "PSISCENE")
            throw std::runtime_error("Invalid scene file");
        int n, nf;
        read(in, &n, 1);
        read(in, &nf, 1);
        if (n < 1 || n > 300000 || nf != 10)
            throw std::runtime_error("Invalid scene");
        std::vector<int2> freq(nf);
        float cal[12];
        read(in, freq.data(), nf);
        read(in, cal, 12);
        std::vector<Geometry> geo(n);
        std::vector<Truth> truth(n), base(n);
        std::vector<float> rh(size_t(n) * 21), rm(size_t(n) * 40);
        read(in, geo.data(), n);
        read(in, base.data(), n);
        CUDA_CHECK(cudaMemcpyToSymbol(MF, freq.data(), nf * sizeof(int2)));
        CUDA_CHECK(cudaMemcpyToSymbol(INVKP, cal, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(TRANS, cal + 9, 3 * sizeof(float)));
        float kp[9] = {1 / cal[0],
                       -cal[1] / (cal[0] * cal[4]),
                       (cal[1] * cal[5] / cal[4] - cal[2]) / cal[0],
                       0,
                       1 / cal[4],
                       -cal[5] / cal[4],
                       0,
                       0,
                       1};
        CUDA_CHECK(cudaMemcpyToSymbol(KP, kp, sizeof(kp)));

        // Precompute fixed bases and allocate reusable device buffers.
        int hf[5] = {1, 16, 32, 40, 48};
        float hb[10 * N] = {};
        for (int k = 0; k < 10; k++)
            for (int x = 0; x < 512; x++) {
                double a = 2 * 3.141592653589793 * hf[k / 2] * (k % 2 ? -1 : 1) * x / 512;
                hb[k * N + x] = float(cos(a) + sin(a));
            }
        CUDA_CHECK(cudaMemcpyToSymbol(HF, hf, sizeof(hf)));
        std::vector<float> denseHB(10 * 512);
        for (int k = 0; k < 10; k++)
            for (int x = 0; x < 512; x++)
                denseHB[k * 512 + x] = hb[k * N + x];
        Device<float> dhb(denseHB.size());
        dhb.upload(denseHB.data());
        Device<Geometry> dg(n);
        dg.upload(geo.data());
        std::vector<Geometry> geo512 = geo;
        for (auto &g : geo512) {
            g.ax = (g.ax - .5f) / 2;
            g.ay = (g.ay - .5f) / 2;
            g.bx = (g.bx - .5f) / 2;
            g.by = (g.by - .5f) / 2;
            g.lc = (g.lc + .5f * g.la + .5f * g.lb) / 2;
        }
        Device<Geometry> dg512(n);
        dg512.upload(geo512.data());
        Device<float> profiles(size_t(n) * 4 * 512);
        Device<Truth> db(n), dt(n);
        db.upload(base.data());
        Device<float> synthH(rh.size()), synthM(rm.size()), drh(rh.size()), drm(rm.size()),
            hc(size_t(n) * 20);
        Device<float2> mc(size_t(n) * 10), coarse(n), uv(n);
        Device<float3> xyz(n);
        Device<int> counts(n);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << prop.name << " pixels=" << n << " centers<=" << o.centers
                  << " neighborhood=" << 2 * o.half + 1 << " init_ms="
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               init)
                         .count()
                  << std::endl;
        std::vector<float2> hostUV(n);
        std::vector<float3> pts(n), gt(n);
        Timer timer;
        int done[2] = {0, 0};
        for (int frame = 0; running && (o.frames == 0 || frame < o.frames); frame++) {
            int m = mode.load();
            const float frameNoise = noiseEnabled.load() ? o.noise : 0.0f;
            bool warmup = done[m] < 3;
            float phase = o.animate ? float(frame) * .03f : 0;
            {
                std::lock_guard<std::mutex> lock(displayMutex);
                caption = std::string(m ? "MSPSI" : "HPSI") + " computing frame " + std::to_string(frame)
                          + " | H: HPSI  M: MSPSI  N: noise";
                repaint = true;
            }
            auto start = std::chrono::steady_clock::now();

            // Generate the current fringe images; capture noise once per frame.
            simulate<<<(n + 255) / 256, 256>>>(dg.p, db.p, dt.p, synthH.p, synthM.p, n, phase,
                                               frameNoise);
            CUDA_CHECK(
                cudaMemcpy(rh.data(), synthH.p, rh.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(
                cudaMemcpy(rm.data(), synthM.p, rm.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(truth.data(), dt.p, n * sizeof(Truth), cudaMemcpyDeviceToHost));
            double synthesis =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();

            // Upload the selected image stack and compute its coefficients and responses.
            Times t;
            auto begin = std::chrono::steady_clock::now();
            timer.start();
            if (m == 0)
                drh.upload(rh.data());
            else
                drm.upload(rm.data());
            timer.stop(t.upload);
            timer.start();
            if (m == 0)
                hartleyCoefficients<<<(n * 20 + 255) / 256, 256>>>(drh.p, hc.p, n);
            else
                fourierCoefficients<<<(n * 10 + 255) / 256, 256>>>(drm.p, mc.p, n, 10);
            timer.stop(t.coeff);

            // Select a correspondence for each camera pixel based on the responses.
            if (m == 0) {
                timer.start();
                hProfiles512<<<(n * 512 + 255) / 256, 256>>>(hc.p, dhb.p, profiles.p, n);
                timer.stop(t.profiles);
            }
            timer.start();
            if (m == 0) {
                hSearch512<<<(n + 127) / 128, 128>>>(profiles.p, dg512.p, coarse.p, counts.p, n, 5);
            } else {
                centerSearch<<<(n + 127) / 128, 128>>>(mc.p, dg.p, coarse.p, counts.p, n, 10,
                                                       o.centers, o.half, .9f);
            }
            timer.stop(t.search);

            // Refine the selected peak to subpixel precision.
            if (m == 0) {
                timer.start();
                hRefine512<<<(n + 127) / 128, 128>>>(profiles.p, coarse.p, uv.p, n, 5);
                timer.stop(t.subpixel);
            }
            if (m == 1) {
                timer.start();
                quadratic2D<<<(n + 127) / 128, 128>>>(mc.p, coarse.p, uv.p, n, 10);
                timer.stop(t.subpixel);
            }

            // Recover 3D coordinates and download the completed reconstruction.
            timer.start();
            triangulate<<<(n + 255) / 256, 256>>>(uv.p, dg.p, xyz.p, n);
            timer.stop(t.triangulation);
            timer.start();
            CUDA_CHECK(cudaMemcpy(hostUV.data(), uv.p, n * sizeof(float2), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pts.data(), xyz.p, n * sizeof(float3), cudaMemcpyDeviceToHost));
            timer.stop(t.download);
            timer.resolve();

            // Measure error against the reference surface outside reconstruction timing.
            t.wall =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                    .count();
            int finite = 0, bad = 0;
            double eu = 0, ez = 0;
            for (int p = 0; p < n; p++) {
                gt[p] = make_float3(truth[p].x, truth[p].y, truth[p].z);
                if (std::isfinite(pts[p].z) && std::isfinite(hostUV[p].x)) {
                    finite++;
                    double e = hypot(hostUV[p].x - truth[p].u, hostUV[p].y - truth[p].v);
                    eu += e * e;
                    ez += pow(pts[p].z - truth[p].z, 2);
                    bad += e > 1;
                }
            }
            eu = sqrt(eu / std::max(finite, 1));
            ez = sqrt(ez / std::max(finite, 1));

            // Publish a complete frame to the viewer.
            {
                std::lock_guard<std::mutex> lock(displayMutex);
                displayed = pts;
                ground = gt;
                std::ostringstream s;
                s << "Truth (left) | " << (m ? "MSPSI" : "HPSI") << " (right) " << std::fixed
                  << std::setprecision(2) << t.wall << " ms | UV RMSE " << eu << " px | Noise "
                  << (frameNoise > 0 ? "ON" : "OFF") << " | H: HPSI  M: MSPSI  N: noise";
                caption = s.str();
                repaint = true;
            }
            double e2e =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            const char *name = m ? "MSPSI" : "HPSI";
            std::cout << name << " frame=" << frame << " noise=" << frameNoise
                      << (warmup ? " warmup" : "") << std::fixed << std::setprecision(3)
                      << " upload=" << t.upload;
            if (m == 0) std::cout << " coeff=" << t.coeff;
            std::cout << " LTC Computation=" << (m ? t.coeff + t.profiles : t.profiles) << " Peak Search=" << t.search
                      << " subpixel=" << t.subpixel << " tri=" << t.triangulation
                      << " download=" << t.download << " total=" << t.wall << " ms e2e=" << e2e
                      << " UV_RMSE=" << eu << " bad=" << bad << std::endl;
            done[m]++;
        }
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        running = false;
        throw;
    }
}

// Console selection must not suspend the reconstruction worker.
struct ConsoleInputMode {
    HANDLE input = GetStdHandle(STD_INPUT_HANDLE);
    DWORD original = 0;
    bool changed = false;
    ConsoleInputMode() {
        if (GetConsoleMode(input, &original))
            changed = SetConsoleMode(input, (original | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE) != 0;
    }
    ~ConsoleInputMode() { if (changed) SetConsoleMode(input, original); }
};

int main(int argc, char **argv) {
    ConsoleInputMode consoleMode;
    try {
        Options o;
        char exePath[MAX_PATH];
        GetModuleFileNameA(nullptr, exePath, MAX_PATH);
        auto home = std::filesystem::path(exePath).parent_path();
        o.input = (home / "scene.bin").string();
        if (!std::filesystem::exists(o.input))
            o.input = (home.parent_path() / "scene.bin").string();
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            auto value = [&]() {
                if (++i >= argc)
                    throw std::runtime_error("Missing argument");
                return std::string(argv[i]);
            };
            if (a == "--input")
                o.input = value();
            else if (a == "--frames")
                o.frames = std::stoi(value());
            else if (a == "--centers")
                o.centers = std::stoi(value());
            else if (a == "--half")
                o.half = std::stoi(value());
            else if (a == "--noise") {
                o.noise = std::stof(value());
                noiseEnabled = o.noise > 0;
                if (o.noise == 0)
                    o.noise = .4f;
            } else if (a == "--headless")
                o.headless = true;
            else if (a == "--static")
                o.animate = false;
            else if (a == "--mode") {
                auto v = value();
                if (v != "hpsi" && v != "mspsi")
                    throw std::runtime_error("Invalid mode");
                mode = v == "hpsi" ? 0 : 1;
            } else
                throw std::runtime_error("Unknown argument: " + a);
        }
        if (!std::isfinite(o.noise) || o.frames < 0 || o.centers < 1 || o.centers > 1024 ||
            o.half < 1 || o.half > 16 || o.noise < 0)
            throw std::runtime_error("Invalid settings");
        if (o.headless) {
            compute(o);
            return 0;
        }

        // Create the OpenGL viewer and start reconstruction on a worker thread.
        WNDCLASSA wc{};
        wc.style = CS_OWNDC;
        wc.lpfnWndProc = wndProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = "PSIClean";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        RegisterClassA(&wc);
        HWND win = CreateWindowA(wc.lpszClassName, "HPSI / MSPSI", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                                 CW_USEDEFAULT, CW_USEDEFAULT, 1280, 800, nullptr, nullptr,
                                 wc.hInstance, nullptr);
        if (!win)
            throw std::runtime_error("Window creation failed");
        HDC dc = GetDC(win);
        PIXELFORMATDESCRIPTOR pf{};
        pf.nSize = sizeof(pf);
        pf.nVersion = 1;
        pf.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        pf.iPixelType = PFD_TYPE_RGBA;
        pf.cColorBits = 24;
        pf.cDepthBits = 24;
        SetPixelFormat(dc, ChoosePixelFormat(dc, &pf), &pf);
        HGLRC rc = wglCreateContext(dc);
        if (!rc || !wglMakeCurrent(dc, rc))
            throw std::runtime_error("OpenGL context failed");
        std::cout << "OpenGL renderer: " << glGetString(GL_RENDERER) << std::endl;
        glEnable(GL_DEPTH_TEST);
        std::atomic<bool> finished{false}, failed{false};
        std::thread worker([&]() {
            try {
                compute(o);
            } catch (...) {
                failed = true;
            }
            finished = true;
        });
        while (running) {
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            if (!running)
                break;
            if (!repaint.exchange(false)) {
                if (finished && o.frames > 0)
                    running = false;
                Sleep(16);
                continue;
            }
            RECT rect;
            GetClientRect(win, &rect);
            glClearColor(.025f, .035f, .055f, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            {
                std::lock_guard<std::mutex> lock(displayMutex);
                SetWindowTextA(win, caption.c_str());
                drawCloud(ground, 0, rect.right / 2, std::max(1L, rect.bottom));
                drawCloud(displayed, rect.right / 2, rect.right / 2, std::max(1L, rect.bottom));
            }
            if (glGetError() != GL_NO_ERROR) {
                std::cerr << "OpenGL rendering error" << std::endl;
                failed = true;
                running = false;
            }
            SwapBuffers(dc);
            if (finished && o.frames > 0)
                running = false;
            Sleep(16);
        }
        worker.join();
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(rc);
        ReleaseDC(win, dc);
        if (IsWindow(win))
            DestroyWindow(win);
        return failed ? 1 : 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
