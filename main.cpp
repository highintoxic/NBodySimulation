#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <random>
#include <algorithm>
#include <cstdlib>

using namespace std;

const double G = 1.0;
const double dt = 0.01;
const double softening = 1e-5;
const double PI = 3.14159265358979323846;

int main(int argc, char *argv[])
{
    int N = 1000;
    int steps = 10000;
    int thread_count = omp_get_max_threads();
    bool write_output = false;

    if (argc > 1)
    {
        N = max(2, atoi(argv[1]));
    }
    if (argc > 2)
    {
        steps = max(1, atoi(argv[2]));
    }
    if (argc > 3)
    {
        thread_count = max(1, atoi(argv[3]));
    }
    if (argc > 4)
    {
        write_output = atoi(argv[4]) != 0;
    }

    omp_set_num_threads(thread_count);

    // 🔥 Structure of Arrays (SoA)
    vector<double> x(N), y(N), vx(N), vy(N), fx(N), fy(N), mass(N);

    // RNG
    mt19937 rng(42);
    uniform_real_distribution<double> angle_dist(0, 2 * PI);
    uniform_real_distribution<double> radius_dist(20, 100);
    uniform_real_distribution<double> noise_dist(-0.2, 0.2);

    // Star
    x[0] = 0;
    y[0] = 0;
    vx[0] = 0;
    vy[0] = 0;
    mass[0] = 1.0;

    // Initialize orbiting bodies
    for (int i = 1; i < N; i++)
    {
        double angle = angle_dist(rng);
        double r = radius_dist(rng);

        x[i] = r * cos(angle);
        y[i] = r * sin(angle);

        double v = sqrt(G * mass[0] / r);
        v *= (1.0 + noise_dist(rng)); // break perfect orbit

        vx[i] = -sin(angle) * v;
        vy[i] = cos(angle) * v;

        mass[i] = 1.0;
    }

    ofstream file;
    if (write_output)
    {
        file.open("output.csv");
    }

    double start = omp_get_wtime();

    for (int step = 0; step < steps; step++)
    {
        // Reset forces
#pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            fx[i] = 0.0;
            fy[i] = 0.0;
        }

        // 🔥 Parallel force computation (NO atomics)
#pragma omp parallel
        {
            vector<double> fx_local(N, 0.0);
            vector<double> fy_local(N, 0.0);

#pragma omp for schedule(dynamic, 10) nowait
            for (int i = 0; i < N; i++)
            {
#pragma omp simd
                for (int j = i + 1; j < N; j++)
                {
                    double dx = x[j] - x[i];
                    double dy = y[j] - y[i];
                    double distSqr = dx * dx + dy * dy + softening;

                    double invDist = 1.0 / sqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;

                    double force = G * mass[i] * mass[j] * invDist3;

                    double fx_ij = dx * force;
                    double fy_ij = dy * force;

                    fx_local[i] += fx_ij;
                    fy_local[i] += fy_ij;

                    fx_local[j] -= fx_ij;
                    fy_local[j] -= fy_ij;
                }
            }

#pragma omp critical
            {
                for (int i = 0; i < N; i++)
                {
                    fx[i] += fx_local[i];
                    fy[i] += fy_local[i];
                }
            }
        }

        // Update positions
#pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            vx[i] += (fx[i] / mass[i]) * dt;
            vy[i] += (fy[i] / mass[i]) * dt;

            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
        }

        // Keep star fixed
        x[0] = 0;
        y[0] = 0;
        vx[0] = 0;
        vy[0] = 0;

        if (write_output)
        {
            for (int i = 0; i < N; i++)
            {
                file << step << "," << x[i] << "," << y[i] << "," << mass[i] << "\n";
            }
        }
    }

    double end = omp_get_wtime();
    cout << "Strategy: OpenMP\n";
    cout << "N: " << N << " Steps: " << steps << "\n";
    cout << "Threads: " << thread_count << "\n";
    cout << "TIME_SECONDS: " << (end - start) << "\n";

    if (file.is_open())
    {
        file.close();
    }
    return 0;
}