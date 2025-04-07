#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iostream>
#include <iterator>
#include <mpi.h>
#include <numeric>

#include "../include/a1-helpers.hpp"

using namespace std;

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
std::vector<std::vector<double>>
initialize_inside_grid(SUBGRID_LOCATION location, int n, int m) {

    int size_x = m + 2;
    int size_y = n + 2;

    std::vector<std::vector<double>> subgrid(n + 2,
                                             std::vector<double>(m + 2, 0.0));

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

void printMatrixBeautifully(const std::vector<std::vector<double>> &matrix) {
    // Handle empty matrix
    if (matrix.empty() || matrix[0].empty()) {
        std::cout << "Empty matrix" << std::endl;
        return;
    }

    // Find maximum width needed for each column
    std::vector<size_t> colWidths(matrix[0].size(), 0);

    for (const auto &row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            // Convert to string with precision of 4 decimal places
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << row[j];
            std::string numStr = oss.str();

            // Remove trailing zeros and decimal point if it's a whole number
            size_t lastNonZero = numStr.find_last_not_of('0');
            if (lastNonZero != std::string::npos &&
                numStr[lastNonZero] == '.') {
                numStr = numStr.substr(0, lastNonZero);
            } else if (lastNonZero != std::string::npos) {
                numStr = numStr.substr(0, lastNonZero + 1);
            }

            colWidths[j] = std::max(colWidths[j], numStr.length());
        }
    }

    // Print the matrix with proper spacing and borders
    const std::string horizontalLine =
        "┌" +
        std::string(2 + std::accumulate(colWidths.begin(), colWidths.end(), 0) +
                        (colWidths.size() - 1) * 3,
                    '-') +
        "┐";
    std::cout << horizontalLine << std::endl;

    for (size_t i = 0; i < matrix.size(); ++i) {
        std::cout << "│ ";
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            // Convert to string with precision of 4 decimal places
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << matrix[i][j];
            std::string numStr = oss.str();

            // Remove trailing zeros and decimal point if it's a whole number
            size_t lastNonZero = numStr.find_last_not_of('0');
            if (lastNonZero != std::string::npos &&
                numStr[lastNonZero] == '.') {
                numStr = numStr.substr(0, lastNonZero);
            } else if (lastNonZero != std::string::npos) {
                numStr = numStr.substr(0, lastNonZero + 1);
            }

            std::cout << std::setw(colWidths[j]) << numStr;

            // Add spacing between columns
            if (j < matrix[i].size() - 1) {
                std::cout << " | ";
            }
        }
        std::cout << " │" << std::endl;
    }

    const std::string bottomLine =
        "└" +
        std::string(2 + std::accumulate(colWidths.begin(), colWidths.end(), 0) +
                        (colWidths.size() - 1) * 3,
                    '-') +
        "┘";
    std::cout << bottomLine << std::endl;
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
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int periods[2] = {false, false};
    int reorder = 0;

    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);

    int rank;
    MPI_Comm_rank(grid_comm, &rank);

    int subgrid_coordinates[2];
    MPI_Cart_coords(grid_comm, rank, 2, subgrid_coordinates);

    // Get neighbor ranks
    int left, right, up, down;
    MPI_Cart_shift(grid_comm, 1, 1, &left, &right);
    MPI_Cart_shift(grid_comm, 0, 1, &up, &down);

    // Get submatrix location
    auto location = get_subgrid_location(up, down, left, right);

    auto U = initialize_inside_grid(location, N / ny, M / nx);
    auto W = initialize_inside_grid(location, N / ny, M / nx);
    // End init

    /*
    auto time_compute_1 =
        chrono::high_resolution_clock::now(); // change to MPI_Wtime()

    int i, j;
    double diffnorm;
    int iteration_count = 0;

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
