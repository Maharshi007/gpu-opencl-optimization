#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    constexpr int N = 1024;

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C(N * N, 0.0f);

    // ------------------------------------------------------------
    // 1. Initialize input matrices.
    // ------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] =
                static_cast<float>((i + j) % 100);

            B[i * N + j] =
                static_cast<float>((i - j + 100) % 100);
        }
    }

    // ------------------------------------------------------------
    // 2. CPU baseline: straightforward matrix multiplication.
    // ------------------------------------------------------------
    const auto start =
        std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < N; ++k) {
                sum +=
                    A[i * N + k] *
                    B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }

    const auto end =
        std::chrono::high_resolution_clock::now();

    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    // ------------------------------------------------------------
    // 3. Calculate checksum.
    // ------------------------------------------------------------
    double checksum = 0.0;

    for (float value : C) {
        checksum += value;
    }

    // ------------------------------------------------------------
    // 4. Print results.
    // ------------------------------------------------------------
    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Matrix size: "
        << N
        << " x "
        << N
        << '\n';

    std::cout
        << "CPU execution time: "
        << elapsed_ms
        << " ms\n";

    std::cout
        << "Checksum: "
        << checksum
        << '\n';

    return 0;
}