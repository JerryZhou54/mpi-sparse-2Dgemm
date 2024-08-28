#include <stdio.h>
#include <mpi.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <string.h>
#include "spmat.h"
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>

bool compare_col (tuples a, tuples b) {
	return a.col < b.col;
}

bool compare_row (tuples a, tuples b) {
	return a.row < b.row;
}

int main(int argc, char* argv[]) {
	if (argc != 5) {
		printf("You're not passing in the correct number of arguments");
		return 1;
	}

	MPI_Init(&argc, &argv);

	MPI_Comm comm = MPI_COMM_WORLD;
	int size;
	MPI_Comm_size(comm, &size);

	int n = atoi(argv[1]);
	float s = atof(argv[2]);
	int pf = atoi(argv[3]);
	char *output_file_name = argv[4];
	int num_rows_per_p = n / sqrt(size);

	float temp;
	int num;

	// unsigned long long C_local[num_rows_per_p][num_rows_per_p];
	std::vector<unsigned long long> C_local(num_rows_per_p*num_rows_per_p);
	std::vector<tuples> A_sparse;
	std::vector<tuples> B_sparse;
	std::vector<tuples> A_orig;
	std::vector<tuples> B_orig;

	// Create 2d torus topology
	MPI_Comm torus_comm;
	int dims[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);
	int periods[2] = {true, true};
	int reorder = true;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &torus_comm);
	// Get rank and coordinate in the current topology
	int rank;
	MPI_Comm_rank(torus_comm, &rank);
	int coords[2];
  MPI_Cart_coords(torus_comm, rank, 2, coords);

	// Define the MPI datatype for tuples
	MPI_Datatype MPI_TUPLE;
  int lengths[3] = { 1, 1, 1 };
	MPI_Aint displacements[3];
	tuples a;
	MPI_Aint base_address;
	MPI_Get_address(&a, &base_address);
	MPI_Get_address(&a.row, &displacements[0]);
	MPI_Get_address(&a.col, &displacements[1]);
	MPI_Get_address(&a.value, &displacements[2]);
	displacements[0] = MPI_Aint_diff(displacements[0], base_address);
	displacements[1] = MPI_Aint_diff(displacements[1], base_address);
	displacements[2] = MPI_Aint_diff(displacements[2], base_address);
	MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_LONG };
	MPI_Type_create_struct(3, lengths, displacements, types, &MPI_TUPLE);
	MPI_Type_commit(&MPI_TUPLE);
	srand(time(NULL) + rank);

	// initialize A and B as well as local C
	// A and B should be 2D partitioned
	// P_ij should hold A[i*n/p:(i+1)*n/p, j*n/p:(j+1)*n/p] and same for B, where p = sqrt(size)
	int start_row = coords[0]*n / sqrt(size);
	int end_row = (coords[0]+1)*n / sqrt(size);
	int start_col = coords[1]*n / sqrt(size);
	int end_col = (coords[1]+1)*n / sqrt(size);
	A_sparse.reserve(num_rows_per_p*num_rows_per_p*s);
	B_sparse.reserve(num_rows_per_p*num_rows_per_p*s);
	if (pf == 1) {
		A_orig.reserve(num_rows_per_p*num_rows_per_p*s);
		B_orig.reserve(num_rows_per_p*num_rows_per_p*s);
	}
	for (int j = start_col; j < end_col; j++) {
		 for (int i = start_row; i < end_row; i++) {
			temp = (float) rand() / (float) RAND_MAX;
			if (temp < s){
				unsigned long num = (unsigned long) rand();
				tuples t;
				t.row = i;
				t.col = j;
				t.value = num;
				A_sparse.emplace_back(t);
				if (pf == 1)
					A_orig.emplace_back(t);
			}
			temp = (float) rand() / (float) RAND_MAX;
			if (temp < s){
				unsigned long num =  (unsigned long) rand();
				tuples t;
				t.row = i;
				t.col = j;
				t.value = num;
				B_sparse.emplace_back(t);
				if (pf == 1)
					B_orig.emplace_back(t);
			}
			C_local[(i-start_row)*num_rows_per_p + j-start_col] = 0;
		}
	}
	A_sparse.shrink_to_fit();
	B_sparse.shrink_to_fit();
	if (pf == 1) {
		A_orig.shrink_to_fit();
		B_orig.shrink_to_fit();
	}
	printf("Rank %d finish initializing A, B, C\n", rank);
	
	double start_time = 0;
	if (rank == 0)
		start_time = MPI_Wtime();
	// Aligning blocks
	enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};
  int neighbours_ranks[4];
	MPI_Cart_shift(torus_comm, 0, 1, &neighbours_ranks[DOWN], &neighbours_ranks[UP]);
 	MPI_Cart_shift(torus_comm, 1, 1, &neighbours_ranks[LEFT], &neighbours_ranks[RIGHT]);
	
	// if (rank == 3) {
	// 	printf("Rank 3 (%d, %d). Sparse A size: %d. Sparse B size: %d\n", coords[0], coords[1], A_sparse.size(), B_sparse.size());
	// }
	printf("Rank %d start aligning A\n", rank);
	int left = 0;
	int right = 0;
	int top = 0;
	int down = 0;
	MPI_Cart_shift(torus_comm, 1, coords[0], &left, &right);
	MPI_Cart_shift(torus_comm, 0, coords[1], &down, &top);

	// Shift A left by i, which means send to left and receive from right
	int send_count = A_sparse.size();
	int recv_count = 0;
	MPI_Send(&send_count, 1, MPI_INT, left, 0, torus_comm);
	MPI_Recv(&recv_count, 1, MPI_INT, right, 0, torus_comm, NULL);
	std::vector<tuples> A_recv(recv_count);
	MPI_Request req;
	MPI_Isend(&A_sparse[0], A_sparse.size(), MPI_TUPLE, left, 1, torus_comm, &req);
	MPI_Recv(&A_recv[0], recv_count, MPI_TUPLE, right, 1, torus_comm, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	A_sparse.assign(A_recv.begin(), A_recv.end());
	A_recv = std::vector<tuples>();
	/**for (int i = 0; i < coords[0]; i++) {
		int send_count = A_sparse.size();
		int recv_count = 0;
		MPI_Send(&send_count, 1, MPI_INT, neighbours_ranks[LEFT], 0, torus_comm);
		MPI_Recv(&recv_count, 1, MPI_INT, neighbours_ranks[RIGHT], 0, torus_comm, NULL);

		std::vector<tuples> A_recv(recv_count);
		MPI_Request req;
		MPI_Isend(&A_sparse[0], A_sparse.size(), MPI_TUPLE, neighbours_ranks[LEFT], 1, torus_comm, &req);
		MPI_Recv(&A_recv[0], recv_count, MPI_TUPLE, neighbours_ranks[RIGHT], 1, torus_comm, NULL);
		MPI_Wait(&req, MPI_STATUS_IGNORE);
		A_sparse.assign(A_recv.begin(), A_recv.end());
		if (i == coords[0] - 1)
			A_recv = std::vector<tuples>();
	}*/

	printf("Rank %d start aligning B\n", rank);
	// Shift B up by j, which means send to up and receive from down
	send_count = B_sparse.size();
	recv_count = 0;
	MPI_Send(&send_count, 1, MPI_INT, top, 2, torus_comm);
	MPI_Recv(&recv_count, 1, MPI_INT, down, 2, torus_comm, NULL);
	std::vector<tuples> B_recv(recv_count);
	MPI_Isend(&B_sparse[0], B_sparse.size(), MPI_TUPLE, top, 3, torus_comm, &req);
	MPI_Recv(&B_recv[0], recv_count, MPI_TUPLE, down, 3, torus_comm, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	B_sparse.assign(B_recv.begin(), B_recv.end());
	B_recv = std::vector<tuples>();
	/**for (int j = 0; j < coords[1]; j++) {
		int send_count = B_sparse.size();
		int recv_count = 0;
		MPI_Send(&send_count, 1, MPI_INT, neighbours_ranks[UP], 2, torus_comm);
		MPI_Recv(&recv_count, 1, MPI_INT, neighbours_ranks[DOWN], 2, torus_comm, NULL);

		std::vector<tuples> B_recv(recv_count);
		MPI_Request req;
		MPI_Isend(&B_sparse[0], B_sparse.size(), MPI_TUPLE, neighbours_ranks[UP], 3, torus_comm, &req);
		MPI_Recv(&B_recv[0], recv_count, MPI_TUPLE, neighbours_ranks[DOWN], 3, torus_comm, NULL);
		MPI_Wait(&req, MPI_STATUS_IGNORE);
		B_sparse.assign(B_recv.begin(), B_recv.end());
		if (j == coords[1] - 1)
			B_recv = std::vector<tuples>();
	}*/

	std::sort(A_sparse.begin(), A_sparse.end(), compare_col);
	std::sort(B_sparse.begin(), B_sparse.end(), compare_row);
	printf("Rank %d finish sorting\n", rank);

	// if (rank == 3) {
	// 	printf("Rank 3 (%d, %d). Sparse A size: %d. Sparse B size: %d\n", coords[0], coords[1], A_sparse.size(), B_sparse.size());
	// }
	// Now we'll do local matrix multiplication
	/**
	for (int i = 0; i < A_sparse.size(); i++) {
		for (int j = 0; j < B_sparse.size(); j++) {
			if (A_sparse[i].col == B_sparse[j].row) {
				int local_i = A_sparse[i].row % (int) (n / sqrt(size));
				int local_j = B_sparse[j].col % (int) (n / sqrt(size));
				C_local[local_i * num_rows_per_p + local_j] += A_sparse[i].value * B_sparse[j].value;
			}	
		} 
	}*/
	int ptr_A = 0;
	int ptr_B = 0;
	while ((ptr_A < A_sparse.size()) && (ptr_B < B_sparse.size())) {
		while ((ptr_A < A_sparse.size()) && (A_sparse[ptr_A].col < B_sparse[ptr_B].row)) {
			ptr_A = ptr_A + 1;
		}
		if (ptr_A >= A_sparse.size())
			break;
		if (A_sparse[ptr_A].col == B_sparse[ptr_B].row) {
			int end_A = ptr_A;
			int end_B = ptr_B;
			// Calculate the value of end_A and end_B
			while (end_A < A_sparse.size()) {
				if (A_sparse[end_A].col == A_sparse[ptr_A].col) {
					end_A = end_A + 1;
				} else {
					break;
				}
			}
			while (end_B < B_sparse.size()) {
				if (B_sparse[end_B].row == B_sparse[ptr_B].row) {
					end_B = end_B + 1;
				} else {
					break;
				}
			}
			// Now we can start O(n^2) operation
			for (int start_B = ptr_B; start_B < end_B; start_B++) {
				for (int start_A = ptr_A; start_A < end_A; start_A++) {
					int local_i = A_sparse[start_A].row % (int) (n / sqrt(size));
					int local_j = B_sparse[start_B].col % (int) (n / sqrt(size));
					C_local[local_i * num_rows_per_p + local_j] += A_sparse[start_A].value * B_sparse[start_B].value;
				}
			}
			ptr_A = end_A;
			ptr_B = end_B;
		} else {
			ptr_B = ptr_B + 1;
		}
	}

	// Now we enter compute and shift phase
	for (int iter = 0; iter < (int) sqrt(size) - 1; iter++) {
		printf("Rank %d(%d,%d), %d\n", rank, coords[0], coords[1], iter);
		int send_count_A = A_sparse.size();
		int recv_count_A = 0;
		MPI_Send(&send_count_A, 1, MPI_INT, neighbours_ranks[LEFT], 4, torus_comm);
		MPI_Recv(&recv_count_A, 1, MPI_INT, neighbours_ranks[RIGHT], 4, torus_comm, NULL);

		// printf("Rank %d Done sending and recving count\n", rank);

		std::vector<tuples> A_recv(recv_count_A);
		MPI_Request req;
		MPI_Isend(&A_sparse[0], send_count_A, MPI_TUPLE, neighbours_ranks[LEFT], 5, torus_comm, &req);
		MPI_Recv(&A_recv[0], recv_count_A, MPI_TUPLE, neighbours_ranks[RIGHT], 5, torus_comm, NULL);
		MPI_Wait(&req, MPI_STATUS_IGNORE);
		A_sparse.assign(A_recv.begin(), A_recv.end());
		A_recv = std::vector<tuples>();

		printf("Rank %d Done shifting A\n", rank);

		int send_count_B = B_sparse.size();
		int recv_count_B = 0;
		MPI_Send(&send_count_B, 1, MPI_INT, neighbours_ranks[UP], 6, torus_comm);
		MPI_Recv(&recv_count_B, 1, MPI_INT, neighbours_ranks[DOWN], 6, torus_comm, NULL);

		std::vector<tuples> B_recv(recv_count_B);
		MPI_Request req2;
		MPI_Isend(&B_sparse[0], send_count_B, MPI_TUPLE, neighbours_ranks[UP], 7, torus_comm, &req2);
		MPI_Recv(&B_recv[0], recv_count_B, MPI_TUPLE, neighbours_ranks[DOWN], 7, torus_comm, NULL);
		MPI_Wait(&req2, MPI_STATUS_IGNORE);
		B_sparse.assign(B_recv.begin(), B_recv.end());
		B_recv = std::vector<tuples>();

		printf("Rank %d Done shifting B\n", rank);
		
		int ptr_A = 0;
		int ptr_B = 0;
		while ((ptr_A < A_sparse.size()) && (ptr_B < B_sparse.size())) {
			while ((ptr_A < A_sparse.size()) && (A_sparse[ptr_A].col < B_sparse[ptr_B].row)) {
				ptr_A = ptr_A + 1;
			}
			if (ptr_A >= A_sparse.size())
				break;
			if (A_sparse[ptr_A].col == B_sparse[ptr_B].row) {
				int end_A = ptr_A;
				int end_B = ptr_B;
				// Calculate the value of end_A and end_B
				while (end_A < A_sparse.size()) {
					if (A_sparse[end_A].col == A_sparse[ptr_A].col) {
						end_A = end_A + 1;
					} else {
						break;
					}
				}
				while (end_B < B_sparse.size()) {
					if (B_sparse[end_B].row == B_sparse[ptr_B].row) {
						end_B = end_B + 1;
					} else {
						break;
					}
				}
				// Now we can start O(n^2) operation
				for (int start_B = ptr_B; start_B < end_B; start_B++) {
					for (int start_A = ptr_A; start_A < end_A; start_A++) {
						int local_i = A_sparse[start_A].row % (int) (n / sqrt(size));
						int local_j = B_sparse[start_B].col % (int) (n / sqrt(size));
						C_local[local_i * num_rows_per_p + local_j] += A_sparse[start_A].value * B_sparse[start_B].value;					
					}
				}
				ptr_A = end_A;
				ptr_B = end_B;
			} else {
				ptr_B = ptr_B + 1;
			}
		}
		/**
		for (int i = 0; i < A_sparse.size(); i++) {
			for (int j = 0; j < B_sparse.size(); j++) {
				if (A_sparse[i].col == B_sparse[j].row) {
					int local_i = A_sparse[i].row % (int) (n / sqrt(size));
					int local_j = B_sparse[j].col % (int) (n / sqrt(size));
					C_local[local_i * num_rows_per_p + local_j] += A_sparse[i].value * B_sparse[j].value;
				}	
			} 
		}*/

		printf("Rank %d done multiplying\n", rank);

	}

	MPI_Barrier(torus_comm);
	if (rank == 0) {
		double end_time = MPI_Wtime();
		printf("Total time taken: %f\n", end_time - start_time);
	}

	if (pf == 1) {
		std::vector<uint64_t> dense_A(n*n, 0);
		std::vector<uint64_t> dense_B(n*n, 0);
		std::vector<uint64_t> dense_C(n*n, 0);

		std::vector<uint64_t> dense_localA(n*n/size, 0);
		std::vector<uint64_t> dense_localB(n*n/size, 0);

		for (int i = 0; i < A_orig.size(); i++) {
			int row = A_orig[i].row % num_rows_per_p;
			int col = A_orig[i].col % num_rows_per_p;
			int index = row*num_rows_per_p + col;
			dense_localA[index] = A_orig[i].value;
		}

		for (int i = 0; i < B_orig.size(); i++) {
			int row = B_orig[i].row % num_rows_per_p;
			int col = B_orig[i].col % num_rows_per_p;
			int index = row*num_rows_per_p + col;
			dense_localB[index] = B_orig[i].value;
		}

		MPI_Gather(&dense_localA[0], n*n/size, MPI_UINT64_T, &dense_A[0], n*n/size, MPI_UINT64_T, 0, torus_comm);
		MPI_Gather(&dense_localB[0], n*n/size, MPI_UINT64_T, &dense_B[0], n*n/size, MPI_UINT64_T, 0, torus_comm);
		MPI_Gather(&C_local[0], n*n/size, MPI_UINT64_T, &dense_C[0], n*n/size, MPI_UINT64_T, 0, torus_comm);	

		// if (rank == 0) {
		// 	for (int i = 0; i < dense_B.size(); i++)
		// 		printf("%d Value:%llu\n", i, dense_B[i]);
		// }
		if (rank == 0) {
			// for (int i = 0; i < A_orig.size(); i++) {
			// 	printf("(%d,%d):%d\n", A_orig[i].row, A_orig[i].col, A_orig[i].value);
			// }
			// Writing the matrix to output file
			FILE *output_file = fopen(output_file_name, "w");
			if (output_file == NULL) {
				perror("Error opening output file");
				return 1;
			}
			// Write the matrix to the output file
			int start_row_addr = 0;
			int start_col_addr = 0;
			int col_offset = 0;
			for (int i = 0; i < n; i++) {
				start_row_addr = int(i/num_rows_per_p) * num_rows_per_p * n + (i % num_rows_per_p)*num_rows_per_p;
				for (int j = 0; j < n; j++) {
					start_col_addr = int(j/num_rows_per_p) * (n*n / size);
					col_offset = j % num_rows_per_p;
					fprintf(output_file, "%llu ", dense_A[start_row_addr+start_col_addr+col_offset]);
				}
				fprintf(output_file, "\n");
			}
			fprintf(output_file, "\n");
			start_row_addr = 0;
			start_col_addr = 0;
			col_offset = 0;
			for (int i = 0; i < n; i++) {
				start_row_addr = int(i/num_rows_per_p) * num_rows_per_p * n + (i % num_rows_per_p)*num_rows_per_p;
				for (int j = 0; j < n; j++) {
					start_col_addr = int(j/num_rows_per_p) * (n*n / size);
					col_offset = j % num_rows_per_p;
					fprintf(output_file, "%llu ", dense_B[start_row_addr+start_col_addr+col_offset]);
				}
				fprintf(output_file, "\n");
			}
			fprintf(output_file, "\n");
			start_row_addr = 0;
			start_col_addr = 0;
			col_offset = 0;
			for (int i = 0; i < n; i++) {
				start_row_addr = int(i/num_rows_per_p) * num_rows_per_p * n + (i % num_rows_per_p)*num_rows_per_p;
				for (int j = 0; j < n; j++) {
					start_col_addr = int(j/num_rows_per_p) * (n*n / size);
					col_offset = j % num_rows_per_p;
					fprintf(output_file, "%llu ", dense_C[start_row_addr+start_col_addr+col_offset]);
				}
				fprintf(output_file, "\n");
			}
			fclose(output_file);
		}


	}

	MPI_Finalize();
	return 0;
}