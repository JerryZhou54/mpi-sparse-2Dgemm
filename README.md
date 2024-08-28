# mpi-sparse-2Dgemm

Overview: This project contains an efficient parallel algorithm to perform sparse matrix multiplication where the matrices are stored as arrays of (row, col, value) tuple

Experimental Setup: The experiments were conducted using the Georgia Tech PACE ICE-cluster.

Program Framework: We generated 'Tuple' data structure to represent nonzero entries in sparse matrices. This data structure is in the form of (int row number, int column number, int entry value), which we call sparse form compared to the dense matrix format. To enable communication of these tuples via MPI, we defined an MPI data type named 'MPI_TUPLE'.

Data Structure and Initialization: We used two 1-D vectors to store the tuples for matrices A and B. A and B were initialized in sparse form, with each entry determined by a randomly generated probability from the range [0,1]. For matrix C, which we initialized in dense form, we used a single 1-D vector with a size of n*n/p, where n is the dimension of the square matrices and p is the number of processors. Matrix C was initialized with all entries set to zero.

Transpose of Matrix B: 
We implemented a Many-to-Many communication primitive to calculate the transpose of matrix B.  Before applying Many-to-Many, we need to calculate the message size for each processor. We used All-to-All to calculate the number of MPI_TUPLE each processor is sending and receiving. We implemented this calculation when we initialized the matrix. To optimize memory usage, we calculated the displacements for the MPI_TUPLE at the destination based on the sending/receiving counts of MPI_TUPLE. Each processor exchanged non-zero entries of its respective n/p×n/p block with every other processor. The n/p×n/p block is determined by the rank of the processor and its distance to the sending/receiving processor, as shown in the 22nd page in lecture slides. This ensured that each processor had n/p rows of matrix A and n/p columns of matrix B ready for multiplication.

Matrix Multiplication Using Shifting:
We get i-th n/p rows of matrix C in the i-th processor by multiplying the i-th n/p rows of matrix A and all columns of matrix B. Therefore, we need to shift the transposed B p times, to make sure every processor gets all blocks of n/p columns of matrix B.
Hence, our algorithm is to shift a block of size n*n/p of matrix B, compute partial sums of the fixed n/p rows and shifted n/p columns in each shift of the ring topology. We accumulated these partial sums through the shifting process. After completing p shifts, the resultant matrix C was fully computed.

Two Pointers in Block Matrix Multiplication
As we need to compute the product of n/p rows and n/p columns in each shift, we decide to use two pointers to speed up the process. First, we sort the block matrix of A and B by the row number and column number correspondingly. Given the sorted two vectors, we can use two pointers to locate the index of the first and last entry of each row of the matrix A and the same for the matrix B. Then we directly implement the dot product and add the partial sum to the corresponding entry in matrix C. 

Output:
To output the matrices, we first converted the sparse form of matrices A and B back into dense form. Using the MPI Gather, we collected the complete matrices at the rank 0 processor.

Bonus:
Instead of using 1d block partitioning, we use 2d block partitioning to distribute matrix A and matrix B, where each processor will store two local matrix with size n/sqrt(p) by n/sqrt(p). Also, we use a 2d-torus topology to embed the processors. Then, we use Cannon's algorithm to perform matrix multiplication. First, we perform the alignment step to ensure that each processor has the correct A and B local matrix to begin calculation. During this step, instead of naively implementing shifting A left by i and shifting B up by j, we directly send the local A and local B matrix to its final destination. Finally, we implement the compute and shift phase.
