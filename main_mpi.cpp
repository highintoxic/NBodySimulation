#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <mpi.h>

using namespace std;

const double G = 1.0;
const double dt = 0.01;
const double softening = 1e-5;
const double PI = 3.14159265358979323846;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 1000;
    int steps = 10000;
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
        write_output = atoi(argv[3]) != 0;
    }

    vector<double> x(N), y(N), vx(N), vy(N), fx(N), fy(N), mass(N);
    vector<double> fx_local(N, 0.0), fy_local(N, 0.0);

    mt19937 rng(42);
    uniform_real_distribution<double> angle_dist(0.0, 2.0 * PI);
    uniform_real_distribution<double> radius_dist(20.0, 100.0);
    uniform_real_distribution<double> noise_dist(-0.2, 0.2);

    x[0] = 0.0;
    y[0] = 0.0;
    vx[0] = 0.0;
    vy[0] = 0.0;
    mass[0] = 1.0;

    for (int i = 1; i < N; ++i)
    {
        const double angle = angle_dist(rng);
        const double r = radius_dist(rng);

        x[i] = r * cos(angle);
        y[i] = r * sin(angle);

        double v = sqrt(G * mass[0] / r);
        v *= (1.0 + noise_dist(rng));

        vx[i] = -sin(angle) * v;
        vy[i] = cos(angle) * v;
        mass[i] = 1.0;
    }

    ofstream file;
    if (rank == 0 && write_output)
    {
        file.open("output_mpi.csv");
    }

    const int chunk = (N + world_size - 1) / world_size;
    const int start_i = rank * chunk;
    const int end_i = min(N, start_i + chunk);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();

    for (int step = 0; step < steps; ++step)
    {
        fill(fx_local.begin(), fx_local.end(), 0.0);
        fill(fy_local.begin(), fy_local.end(), 0.0);

        for (int i = start_i; i < end_i; ++i)
        {
            for (int j = i + 1; j < N; ++j)
            {
                const double dx = x[j] - x[i];
                const double dy = y[j] - y[i];
                const double dist_sqr = dx * dx + dy * dy + softening;

                const double inv_dist = 1.0 / sqrt(dist_sqr);
                const double inv_dist3 = inv_dist * inv_dist * inv_dist;
                const double force = G * mass[i] * mass[j] * inv_dist3;

                const double fx_ij = dx * force;
                const double fy_ij = dy * force;

                fx_local[i] += fx_ij;
                fy_local[i] += fy_ij;

                fx_local[j] -= fx_ij;
                fy_local[j] -= fy_ij;
            }
        }

        MPI_Allreduce(fx_local.data(), fx.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(fy_local.data(), fy.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < N; ++i)
        {
            vx[i] += (fx[i] / mass[i]) * dt;
            vy[i] += (fy[i] / mass[i]) * dt;

            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
        }

        x[0] = 0.0;
        y[0] = 0.0;
        vx[0] = 0.0;
        vy[0] = 0.0;

        if (rank == 0 && write_output)
        {
            for (int i = 0; i < N; ++i)
            {
                file << step << "," << x[i] << "," << y[i] << "," << mass[i] << "\n";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end_time = MPI_Wtime();

    if (rank == 0)
    {
        cout << "Strategy: MPI\n";
        cout << "N: " << N << " Steps: " << steps << "\n";
        cout << "Processes: " << world_size << "\n";
        cout << "TIME_SECONDS: " << (end_time - start_time) << "\n";
        if (file.is_open())
        {
            file.close();
        }
    }

    MPI_Finalize();
    return 0;
}
