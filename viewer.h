#pragma once

std::atomic<bool> running{true};
std::atomic<bool> repaint{true};
std::atomic<int> mode{0};
std::atomic<bool> noiseEnabled{false};
std::mutex displayMutex;
std::vector<float3> displayed, ground;
std::string caption = "Starting CUDA...";
float angleX = 15, angleY = 0, zoom = 1;
bool dragging = false;
int lastX, lastY;
// Handle method selection, noise toggling and camera controls.
LRESULT CALLBACK wndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_PAINT: {
        PAINTSTRUCT ps;
        BeginPaint(h, &ps);
        EndPaint(h, &ps);
        repaint = true;
        return 0;
    }
    case WM_CLOSE:
        running = false;
        DestroyWindow(h);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (w == 'H')
            mode = 0;
        if (w == 'M')
            mode = 1;
        if (w == 'N' && !(l & (1LL << 30)))
            noiseEnabled = !noiseEnabled.load();
        if (w == VK_ESCAPE)
            SendMessage(h, WM_CLOSE, 0, 0);
        return 0;
    case WM_LBUTTONDOWN:
        dragging = true;
        lastX = short(LOWORD(l));
        lastY = short(HIWORD(l));
        SetCapture(h);
        return 0;
    case WM_LBUTTONUP:
        dragging = false;
        ReleaseCapture();
        return 0;
    case WM_MOUSEMOVE:
        if (dragging) {
            int x = short(LOWORD(l)), y = short(HIWORD(l));
            repaint = true;
            angleY += (x - lastX) * .4f;
            angleX += (y - lastY) * .4f;
            lastX = x;
            lastY = y;
        }
        return 0;
    case WM_MOUSEWHEEL:
        repaint = true;
        zoom *= GET_WHEEL_DELTA_WPARAM(w) > 0 ? 1.1f : .91f;
        return 0;
    }
    return DefWindowProc(h, m, w, l);
}
// Draw the reference and reconstructed clouds in the same coordinate frame.
void drawCloud(const std::vector<float3> &pts, int x, int width, int height) {
    glViewport(x, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = double(width) / height;
    glOrtho(-180 * aspect, 180 * aspect, -180, 180, -2000, 2000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(zoom, zoom, zoom);
    glRotatef(angleX, 1, 0, 0);
    glRotatef(angleY, 0, 1, 0);
    if (pts.empty())
        return;
    double cx = 0, cy = 0, cz = 0;
    int n = 0;
    for (auto p : ground)
        if (std::isfinite(p.z)) {
            cx += p.x;
            cy += p.y;
            cz += p.z;
            n++;
        }
    if (!n)
        return;
    glTranslatef(float(-cx / n), float(-cy / n), float(-cz / n));
    glPointSize(2.0f);
    std::vector<float3> colors(pts.size());
    for (size_t i = 0; i < pts.size(); i++) {
        float t = std::clamp((pts[i].z - 460.f) / 140.f, 0.f, 1.f);
        // Use the same fixed depth range for both clouds (460 to 600 mm).
        colors[i] = make_float3(std::clamp(1.5f - fabsf(4*t - 3), 0.f, 1.f),
                                std::clamp(1.5f - fabsf(4*t - 2), 0.f, 1.f),
                                std::clamp(1.5f - fabsf(4*t - 1), 0.f, 1.f));
    }
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_FLOAT, sizeof(float3), colors.data());
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(float3), pts.data());
    glDrawArrays(GL_POINTS, 0, int(pts.size()));
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}


