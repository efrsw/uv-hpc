#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <stdexcept>
#include <string>

#include "../include/a1-helpers.hpp"

using namespace std;

int main(int argc, char **argv) {
    int max_iterations = 2000;
    double epsilon = 1.0e-3;

    // default values for M rows and N columns
    int N = 42;
    int M = 42;

    // size of 2D mesh - only used for your parallel version
    int nx = 6, ny = 7;

    process_input(argc, argv, N, M, nx, ny, max_iterations, epsilon);

    // Note that you need to initialize MPI first
    int numprocs, rank;
    int tag = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype row_type, col_type, col_matrix_type, row_2_type;

    MPI_Type_vector(1, 4, 4, MPI_DOUBLE, &row_type);
    MPI_Type_vector(4, 1, 4, MPI_DOUBLE, &col_type);
    MPI_Type_vector(1, 1, 1, MPI_DOUBLE, &row_2_type);
    MPI_Type_create_hvector(4, 1, sizeof(double), col_type, &col_matrix_type);

    MPI_Type_commit(&row_type);
    MPI_Type_commit(&row_2_type);
    MPI_Type_commit(&col_type);
    MPI_Type_commit(&col_matrix_type);

    if (rank == 0) {
        double matrix[4][4] = {{00, 01, 02, 03},
                               {10, 11, 12, 13},
                               {20, 21, 22, 23},
                               {30, 31, 32, 33}};

        MPI_Send(matrix, 4, row_2_type, 1, tag, MPI_COMM_WORLD);
    } else if (rank == 1) {
        double matrix[4][4] = {};
        MPI_Status recv_status;
        MPI_Recv(matrix, 1, col_matrix_type, 0, 0, MPI_COMM_WORLD,
                 &recv_status);
        print_double_array(matrix, 4, 4, 0);
    }

    /*
    auto time_compute_1 =
        chrono::high_resolution_clock::now(); // change to MPI_Wtime()

    int i, j;
    double diffnorm;
    int iteration_count = 0;

    // You can also use Mat data struct (see helpers), in which case you
    // allocate like: Mat U(M, N) and then use it like a regular 2D array

    // TODO: use local sizes with MPI, i.e., recalculate M and N
    auto U = Mat(M, N);
    // TODO: same as with the U matrix
    auto W = Mat(M, N);

    // Init & Boundary
    for (j = 0; j < M; ++j) {
        for (i = 0; i < N; ++i) {
            W[j][i] = U[j][i] = 0.0;
        }

        W[j][0] = U[j][0] = 20.0;
        W[j][N - 1] = U[j][N - 1] = 40.0;
    }

    for (i = 0; i < N; ++i) {
        W[M - 1][i] = U[M - 1][i] = 100.0;
    }
    // End init

    iteration_count = 0;
    do {
        iteration_count++;
        diffnorm = 0.0;

        // Compute new values (but not on boundary)
        // you need to adjust for-loop start and end values
        {
            for (j = 1; j < M - 1; ++j) {
                for (i = 1; i < N - 1; ++i) {
                    W[j][i] = (U[j][i + 1] + U[j][i - 1] + U[j + 1][i] +
                               U[j - 1][i]) *
                              0.25;
                    diffnorm += (W[j][i] - U[j][i]) * (W[j][i] - U[j][i]);
                }
            }

            // Only transfer the interior points
            for (j = 1; j < M - 1; ++j)
                for (i = 1; i < N - 1; ++i)
                    U[j][i] = W[j][i];
        }

        diffnorm = sqrt(diffnorm); // all processes need to know when to stop
    } while (epsilon <= diffnorm && iteration_count < max_iterations);

    auto time_compute_2 =
        chrono::high_resolution_clock::now(); // change to MPI_Wtime()

    // TODO for MPI: collect all local parts of the U matrix, and save it to
    // another "big" matrix That has the same size the whole size: like
    // bigU(M,N) For verification use output data file

    auto time_io_1 = chrono::high_resolution_clock::now();

    std::ofstream ofs("heat-" + to_string(M) + "x" + to_string(N) + ".txt",
                      std::ofstream::out);
    ofs << N << "\n" << M << "\n";

    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < N; ++i) {
            ofs << std::fixed << std::setfill(' ') << std::right
                << std::setw(11) << setprecision(6) << U[j][i] << " ";
        }
        // ofs << "\n";
    }
    ofs.close();

    auto time_io_2 = chrono::high_resolution_clock::now();

    // print time measurements
    cout << "Computed in " << iteration_count << " iterations and "
         << chrono::duration<double>(time_compute_2 - time_compute_1).count()
         << " seconds." << endl;
    cout << "IO time: "
         << chrono::duration<double>(time_io_2 - time_io_1).count()
         << " seconds." << endl;
    cout
        << "Total time: "
        << chrono::duration<double>(time_io_2 - time_io_1).count() +
               chrono::duration<double>(time_compute_2 - time_compute_1).count()
        << " seconds." << endl;

    */

    // do not forget to call MPI_Finalize()
    MPI_Finalize();

    return EXIT_SUCCESS;
}
