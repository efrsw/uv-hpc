#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <numeric>

#include "../include/a1-helpers.hpp"

/**
 * Position of subgrid inside topology
 */
enum SUBGRID_LOCATION {
    TOP_LEFT = 0,
    TOP_EDGE = 1,
    TOP_RIGHT = 2,
    RIGHT_EDGE = 3,
    BOTTOM_RIGHT = 4,
    BOTTOM_EDGE = 5,
    LEFT_EDGE = 6,
    BOTTOM_LEFT = 7,
    INTERIOR = 8
};

SUBGRID_LOCATION get_subgrid_location(int up, int down, int left, int right) {
    SUBGRID_LOCATION location;

    if (left != MPI_PROC_NULL && right != MPI_PROC_NULL &&
        up != MPI_PROC_NULL && down != MPI_PROC_NULL) {
        location = INTERIOR;
    } else if (up == MPI_PROC_NULL && left == MPI_PROC_NULL) {
        location = TOP_LEFT;
    } else if (up == MPI_PROC_NULL && right == MPI_PROC_NULL) {
        location = TOP_RIGHT;
    } else if (down == MPI_PROC_NULL && left == MPI_PROC_NULL) {
        location = BOTTOM_LEFT;
    } else if (down == MPI_PROC_NULL && right == MPI_PROC_NULL) {
        location = BOTTOM_RIGHT;
    } else if (up == MPI_PROC_NULL) {
        location = TOP_EDGE;
    } else if (down == MPI_PROC_NULL) {
        location = BOTTOM_EDGE;
    } else if (left == MPI_PROC_NULL) {
        location = LEFT_EDGE;
    } else if (right == MPI_PROC_NULL) {
        location = RIGHT_EDGE;
    }

    return location;
}

/**
 * Initialized grid based on its position inside topology
 */
Mat initialize_inside_grid(SUBGRID_LOCATION location, int n, int m) {

    int size_x = m + 2;
    int size_y = n + 2;

    Mat subgrid(n + 2, m + 2, 0);

    switch (location) {
    case TOP_LEFT:
    case LEFT_EDGE:
        for (auto i = 1; i < size_y - 1; i++) {
            subgrid[i][1] = 20;
        }
        break;
    case TOP_RIGHT:
    case RIGHT_EDGE:
        for (auto i = 1; i < size_y - 1; i++) {
            subgrid[i][size_x - 2] = 40;
        }
        break;
    case BOTTOM_RIGHT:
        for (auto i = 1; i < size_y - 1; i++) {
            subgrid[i][size_x - 2] = 40;
        }
        std::fill(&subgrid[size_y - 2][1], &subgrid[size_y - 2][size_x - 1],
                  100.0);
        break;
    case BOTTOM_LEFT:
        for (auto i = 1; i < size_y - 1; i++) {
            subgrid[i][1] = 20;
        }
        std::fill(&subgrid[size_y - 2][1], &subgrid[size_y - 2][size_x - 1],
                  100.0);
        break;
    case BOTTOM_EDGE:
        std::fill(&subgrid[size_y - 2][1], &subgrid[size_y - 2][size_x - 1],
                  100.0);
        break;
    case TOP_EDGE:
    case INTERIOR:
        break;
    }

    return subgrid;
}

int main(int argc, char **argv) {
    int max_iterations = 2000;
    double epsilon = 1.0e-3;

    // default values for N rows and M columns
    int N = 6;
    int M = 12;

    // size of 2D mesh - only used for your parallel version
    int nx = 2, ny = 3;

    // process_input(argc, argv, N, M, nx, ny, max_iterations, epsilon);

    // Init & Boundary
    // Note that you need to initialize MPI first
    int comm_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int dims[2] = {0, 0};
    MPI_Dims_create(comm_size, 2, dims);

    int periods[2] = {false, false};
    int reorder = 0;

    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);

    int rank;
    MPI_Comm_rank(grid_comm, &rank);

    int subgrid_coordinates[2];
    MPI_Cart_coords(grid_comm, rank, 2, subgrid_coordinates);

    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, 1, &left, &right);
    MPI_Cart_shift(grid_comm, 0, 1, &up, &down);

    auto location = get_subgrid_location(up, down, left, right);

    // Size of subgrids
    int dx, dy;
    dx = M / nx;
    dy = N / ny;

    auto U = initialize_inside_grid(location, dy, dx);
    auto W = initialize_inside_grid(location, dy, dx);

    // Total size of subgrids (with halo regions)
    int bx, by;
    bx = dx + 2;
    by = dy + 2;

    auto size = bx * by;

    // Subarray size
    int global_size[] = {N, M};
    int local_size[] = {dy, dx};
    int global_start[] = {0, 0};

    MPI_Datatype row_type, col_type;
    MPI_Type_contiguous(dx, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    MPI_Type_vector(dy, 1, bx, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Datatype subarray;
    MPI_Type_create_subarray(2, global_size, local_size, global_start,
                             MPI_ORDER_C, MPI_DOUBLE, &subarray);
    MPI_Type_commit(&subarray);

    MPI_Datatype matrix_row;
    // Take every element except halo region (last element)
    MPI_Type_vector(1, bx - 1, bx, MPI_DOUBLE, &matrix_row);

    MPI_Win win;
    MPI_Win_create(&U[0][0], sizeof(double) * size, sizeof(double),
                   MPI_INFO_NULL, grid_comm, &win);
    // End init

    auto time_compute = MPI_Wtime(); // change to MPI_Wtime()

    double diffnorm;
    int iteration_count = 0;

    iteration_count = 0;
    do {
        iteration_count++;
        diffnorm = 0.0;

        // Halo exchange
        MPI_Win_fence(0, win);
        if (left != MPI_PROC_NULL) {
            // End of the second row is the start of the halo region
            MPI_Aint disp = 2 * bx - 1;
            MPI_Put(&U[1][1], 1, col_type, left, disp, 1, col_type, win);
        }

        if (right != MPI_PROC_NULL) {
            // Beginning of the second row is where the halo starts
            MPI_Aint disp = bx;
            MPI_Put(&U[1][dx], 1, col_type, right, disp, 1, col_type, win);
        }

        if (up != MPI_PROC_NULL) {
            // Halo starts at the second element of the last row
            MPI_Aint disp = (by - 1) * bx + 1;
            MPI_Put(&U[1][1], 1, row_type, up, disp, 1, row_type, win);
        }

        if (down != MPI_PROC_NULL) {
            // Halo starts at the second element of the firs row
            MPI_Aint disp = 1;
            MPI_Put(&U[dy][1], 1, row_type, down, disp, 1, row_type, win);
        }
        MPI_Win_fence(0, win);


        int ccoords[2];
        MPI_Cart_coords(grid_comm, rank, 2, ccoords);

        // Compute new values (but not on boundary)
        // you need to adjust for-loop start and end values
        for (auto j = 1; j < by - 1; j++) {
            for (auto i = 1; i < bx - 1; i++) {
                W[j][i] =
                    (U[j][i + 1] + U[j][i - 1] + U[j + 1][i] + U[j - 1][i]) *
                    0.25;
                diffnorm += (W[j][i] - U[j][i]) * (W[j][i] - U[j][i]);
            }
        }

        // Only transfer the interior points
        for (auto j = 1; j < by - 1; ++j)
            for (auto i = 1; i < bx - 1; ++i)
                U[j][i] = W[j][i];

        double global_diffnorm;
        MPI_Allreduce(&diffnorm, &global_diffnorm, 1, MPI_DOUBLE, MPI_SUM,
                      grid_comm);
        diffnorm =
            sqrt(global_diffnorm); // all processes need to know when to stop
    } while (epsilon <= diffnorm && iteration_count < max_iterations);

    std::cout << U.to_string() << std::endl;
    time_compute += MPI_Wtime();

    // TODO for MPI: collect all local parts of the U matrix, and save it to
    // another "big" matrix That has the same size the whole size: like
    // bigU(M,N) For verification use output data file

    /*
    if (rank == 0) {
        auto time_io_1 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<double>> global_U(N,
                                                  std::vector<double>(M, 0.0));

        MPI_Gather(MPI_IN_PLACE, 1, subarray, global_U.data(), 1, subarray, 0,
    grid_comm);

        // Write data
        std::ofstream ofs("heat-" + std::to_string(M) + "x" +
                              std::to_string(N) + ".txt",
                          std::ofstream::out);
        ofs << N << "\n" << M << "\n";

        for (int j = 0; j < M; ++j) {
            for (int i = 0; i < N; ++i) {
                ofs << std::fixed << std::setfill(' ') << std::right
                    << std::setw(11) << std::setprecision(6) << global_U[j][i]
                    << " ";
            }
            // ofs << "\n";
        }
        ofs.close();

        auto time_io_2 = std::chrono::high_resolution_clock::now();

        // print time measurements
        std::cout << std::format("Computed in {} iterations and {} seconds.",
                                 iteration_count, time_compute)
                  << std::endl;
        std::cout
            << "IO time: "
            << std::chrono::duration<double>(time_io_2 - time_io_1).count()
            << " seconds." << std::endl;
        std::cout
            << "Total time: "
            << std::chrono::duration<double>(time_io_2 - time_io_1).count() +
                   std::chrono::duration<double>(time_compute - time_compute)
                       .count()
            << " seconds." << std::endl;

    } else {
        MPI_Gather(&U[1][1], dy, matrix_row, nullptr, 0, MPI_DATATYPE_NULL, 0,
    grid_comm);
    }
    */

    // do not forget to call MPI_Finalize()
    MPI_Type_free(&row_type);
    MPI_Type_free(&col_type);
    MPI_Type_free(&subarray);
    MPI_Finalize();

    return EXIT_SUCCESS;
}
