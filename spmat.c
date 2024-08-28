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
	int num_rows_per_p = n / size;
	// unsigned long long C[num_rows_per_p][n];
	std::vector<unsigned long long> C(num_rows_per_p*n);

	float temp;
	int num;

	int counts_send[size];
	std::fill(counts_send, counts_send + size, 0);
	std::vector<tuples> A2d;
	std::vector<tuples> B2d;
	int block_size = int(n/size);

	// Create ring topology
	MPI_Comm ring_comm;
	const int dims[1] = {size};
	const int periods[1] = {true};
	MPI_Cart_create(comm, 1, dims, periods, true, &ring_comm);

	// Get rank in the current topology
	int rank;
	MPI_Comm_rank(ring_comm, &rank);

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
	printf("Rank %d start initializing matrix\n", rank);
	// initialize A2d and B2d
	A2d.reserve(n*num_rows_per_p*s);
	B2d.reserve(n*num_rows_per_p*s);
	for (int j = 0; j < n; j++) {
		 for (int i = rank*num_rows_per_p; i < (rank+1)*num_rows_per_p; i++) {
			temp = (float) rand() / (float) RAND_MAX;
			if (temp < s){
				unsigned long num = (unsigned long) rand();
				tuples t;
				t.row = i;
				t.col = j;
				t.value = num;
				A2d.emplace_back(t);
			}
			temp = (float) rand() / (float) RAND_MAX;
			if (temp < s){
				unsigned long num =  (unsigned long) rand();
				tuples t;
				t.row = i;
				t.col = j;
				t.value = num;
				B2d.emplace_back(t);
				counts_send[int(j/block_size)] += 1;
				
			}
			C[(i%num_rows_per_p) * n + j] = 0;
		}
	}
	A2d.shrink_to_fit();
	B2d.shrink_to_fit();
	
	printf("Rank %d finish initializing matrix\n", rank);
	double start_time = 0;
	if (rank == 0)
		start_time = MPI_Wtime();
	// Perform transpose
	int counts_recv[size];
	MPI_Alltoall(counts_send, 1, MPI_INT, counts_recv, 1, MPI_INT, ring_comm);

	int displacements_send[size];
	int displacements_recv[size];
	for (int i = 0; i < size; i++) {
		if (i == 0) {
			displacements_send[i] = 0;
			displacements_recv[i] = 0;
		} else {
			displacements_send[i] = displacements_send[i-1] + counts_send[i-1];
			displacements_recv[i] = displacements_recv[i-1] + counts_recv[i-1];
		}
	}

	int count = 0;
	for (int i = 0; i < size; i++) {
		count = count + counts_recv[i];
	}
	std::vector<tuples> B_T(count);
	MPI_Alltoallv(&B2d[0], counts_send, displacements_send, MPI_TUPLE, &B_T[0], counts_recv, displacements_recv, MPI_TUPLE, ring_comm);
	
	std::sort(A2d.begin(), A2d.end(), compare_col);
	std::sort(B_T.begin(), B_T.end(), compare_row);
	printf("Rank %d finish sorting\n", rank);
	int ptr_A = 0;
	int ptr_B = 0;
	while ((ptr_A < A2d.size()) && (ptr_B < B_T.size())) {
		while ((ptr_A < A2d.size()) && (A2d[ptr_A].col < B_T[ptr_B].row)) {
			ptr_A = ptr_A + 1;
		}
		if (ptr_A >= A2d.size())
			break;
		if (A2d[ptr_A].col == B_T[ptr_B].row) {
			int end_A = ptr_A;
			int end_B = ptr_B;
			// Calculate the value of end_A and end_B
			while (end_A < A2d.size()) {
				if (A2d[end_A].col == A2d[ptr_A].col) {
					end_A = end_A + 1;
				} else {
					break;
				}
			}
			while (end_B < B_T.size()) {
				if (B_T[end_B].row == B_T[ptr_B].row) {
					end_B = end_B + 1;
				} else {
					break;
				}
			}
			// Now we can start O(n^2) operation
			for (int start_B = ptr_B; start_B < end_B; start_B++) {
				for (int start_A = ptr_A; start_A < end_A; start_A++) {
					C[(A2d[start_A].row % block_size)*n + B_T[start_B].col] += A2d[start_A].value * B_T[start_B].value;
				}
			}
			ptr_A = end_A;
			ptr_B = end_B;
		} else {
			ptr_B = ptr_B + 1;
		}
	}
	/**for (int i = 0; i < A2d.size(); i++) {
		for (int j = 0; j < B_T.size(); j++) {
			if (A2d[i].col == B_T[j].row) {
				C[(A2d[i].row % block_size)*n + B_T[j].col] += static_cast<unsigned long long>(A2d[i].value) * B_T[j].value;
			}	
		} 
	}*/
	
	int left, right;
	MPI_Cart_shift(ring_comm, 0, 1, &left, &right);

	for (int i = 0; i < size - 1; i++) {
		// Sender
		int send_count = B_T.size();
		int recv_count;
		MPI_Status stat;
		MPI_Send(&send_count, 1, MPI_INT, right, 0, ring_comm);
		MPI_Recv(&recv_count, 1, MPI_INT, left, 0, ring_comm, &stat);
		std::vector<tuples> B_recv(recv_count);
		MPI_Request req;
		MPI_Isend(&B_T[0], send_count, MPI_TUPLE, right, 0, ring_comm, &req);
		// printf("send_count: %d, rank: %d, round: %d\n", send_count, rank, i);
		// printf("recv_count: %d, rank: %d, round: %d\n", recv_count, rank, i);
		MPI_Recv(&B_recv[0], recv_count, MPI_TUPLE, left, 0, ring_comm, &stat);
		MPI_Wait(&req, MPI_STATUS_IGNORE);
		B_T.assign(B_recv.begin(), B_recv.end());
		B_recv = std::vector<tuples>();

		int ptr_A = 0;
		int ptr_B = 0;
		while ((ptr_A < A2d.size()) && (ptr_B < B_T.size())) {
			while ((ptr_A < A2d.size()) && (A2d[ptr_A].col < B_T[ptr_B].row)) {
				ptr_A = ptr_A + 1;
			}
			if (ptr_A >= A2d.size())
				break;
			if (A2d[ptr_A].col == B_T[ptr_B].row) {
				int end_A = ptr_A;
				int end_B = ptr_B;
				// Calculate the value of end_A and end_B
				while (end_A < A2d.size()) {
					if (A2d[end_A].col == A2d[ptr_A].col) {
						end_A = end_A + 1;
					} else {
						break;
					}
				}
				while (end_B < B_T.size()) {
					if (B_T[end_B].row == B_T[ptr_B].row) {
						end_B = end_B + 1;
					} else {
						break;
					}
				}
				// Now we can start O(n^2) operation
				for (int start_B = ptr_B; start_B < end_B; start_B++) {
					for (int start_A = ptr_A; start_A < end_A; start_A++) {
						C[(A2d[start_A].row % block_size)*n + B_T[start_B].col] += A2d[start_A].value * B_T[start_B].value;
					}
				}
				ptr_A = end_A;
				ptr_B = end_B;
			} else {
				ptr_B = ptr_B + 1;
			}
		}
		/**for (int i = 0; i < A2d.size(); i++) {
			for (int j = 0; j < B_T.size(); j++) {
				if (A2d[i].col == B_T[j].row) {
					C[(A2d[i].row % block_size)*n + B_T[j].col] += static_cast<unsigned long long>(A2d[i].value) * B_T[j].value;				
				}	
			} 
		}*/
		
	}

	MPI_Barrier(ring_comm);
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

		for (int i = 0; i < A2d.size(); i++) {
			int row = A2d[i].row%num_rows_per_p;
			int col = A2d[i].col;
			int index = row*n + col;
			dense_localA[index] = A2d[i].value;
		}
		for (int i = 0; i < B2d.size(); i++) {
			int row = B2d[i].row%num_rows_per_p;
			int col = B2d[i].col;
			int index = row*n + col;
			dense_localB[index] = B2d[i].value;
		}

		MPI_Gather(&dense_localA[0], n*n/size, MPI_UINT64_T, &dense_A[0], n*n/size, MPI_UINT64_T, 0, ring_comm);
		MPI_Gather(&dense_localB[0], n*n/size, MPI_UINT64_T, &dense_B[0], n*n/size, MPI_UINT64_T, 0, ring_comm);
		MPI_Gather(&C[0], n*n/size, MPI_UINT64_T, &dense_C[0], n*n/size, MPI_UINT64_T, 0, ring_comm);

		if (rank == 0) {
			// Writing the matrix to output file
			FILE *output_file = fopen(output_file_name, "w");
			if (output_file == NULL) {
				perror("Error opening output file");
				return 1;
			}

			// Write the matrix to the output file
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					fprintf(output_file, "%llu ", dense_A[i*n+j]);
				}
				fprintf(output_file, "\n");
			}
			fprintf(output_file, "\n");
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					fprintf(output_file, "%llu ", dense_B[i*n+j]);
				}
				fprintf(output_file, "\n");
			}
			fprintf(output_file, "\n");
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					fprintf(output_file, "%llu ", dense_C[i*n+j]);
				}
				fprintf(output_file, "\n");
			}

			fclose(output_file);
		}
	}

	MPI_Finalize();
	return 0;

}