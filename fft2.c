#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

#define PAIR 2
#define HALF 2

#define LOWER 1
#define UPPER 2

#define SWAP(a,b) tempr=(a); (a)=(b); (b)=tempr

/**
 * Binary Exchange FFT code.
 *
 * By Patrick Hudgell
 * and Uvin Abeysinghe
 *
 * This program implements the ideas behind Binary exchange, to
 * parallelise the DIT-FFT.
 *
 * One of the main ideas behind binary exchange is the fact that not every
 * FFT butterfly requires message passing to be calculated. Infact quite
 * a number of the butterflies can be calculated locally.
 *
 * Hence this algorithm is based upon two rounds of calculations.
 * The first round is for the first log(n/p) iterations of the butterfly
 * calculations, where it is possible to calculate all the butterflies
 * using local data only.
 * The remaining log p iterations involve the use of message passing
 * to calcuate the butterflies with the data that is not locally stored.
 *
 * Due to the fact that there is symmetry between all the nodes, the
 * array of values in one node all go to the same node, when a butterfly
 * involving communication is to be calculated. Hence the use of short
 * messages are avoided.
 *
 **/

typedef long int ULONG;
double start;
double end;

/** Performs DIT FFT on data contained locally. Necessary for round 1, were
 *  all values are stored in local memory */
void processLocal(int round1Iterations, int localValueCount, double complex localValues[]) {
	for(ULONG r1count = 0; r1count < round1Iterations; r1count++) {
		ULONG step = powl(2, r1count+1);
		for(ULONG butPos = 0; butPos < localValueCount; butPos += step) {
			ULONG kCount = 0;			
			for (ULONG idx = butPos; idx < butPos + (step/2); idx++) {
			    double complex t
			        = cexp((-I * (2*M_PI) * kCount) / step)
			        * localValues[idx + (step/2)];

			    localValues[idx + (step/2)] = localValues[idx] - t;
			    localValues[idx] = localValues[idx] + t;
			    kCount += 1;							
			}
		}
	}
}


/** Rearranges data, so the data is in the bit shifted order
 *  Necessary for FFT.
 *  Adapted from Simple Fast Fourier Transform Algo's in C.
 *  http://www.guitarscience.net/papers/fftalg.pdf
 */
void bitShift(ULONG size, double complex data[]) {
	double complex *dataPtr;
	double complex tempr;
	dataPtr = &data[0] - 1;
	
	int m = 0;
	int j = 1;
	int n = size;
	
	for (int i = 1; i < n; i++) {
		if (j > i) {
			SWAP(dataPtr[j], dataPtr[i]);
		}

		m = n >> 1;

		while (m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}
}

/**
 * Returns the colour of the node, depending on if it is in the
 * odd half or the even half of the FFT. 
 * This is used for distributing the correct bit-reversed values
 * to each node.
 * */
int getColour(int world_rank, int world_size) {
	if (world_rank % world_size < world_size / 2) {
		return UPPER;
	}
	return LOWER;
}

void printResult(ULONG count, double complex allValues[]) {
	for(ULONG i = 0; i < count; i++) {
		printf("X[%lu] is %f %f\n", i, creal(allValues[i]), cimag(allValues[i]));
	}

}

void processSequential(ULONG count) {
	double complex *complexArr = malloc(sizeof(double complex) * count);

	double realCompoent;
	double complexCompoent;
	double complex complexSum;
	for (ULONG i = 0; i < count; i++) {
		scanf("%lf %lf", &realCompoent, &complexCompoent);
		complexSum = realCompoent + (complexCompoent * I);
		complexArr[i] = complexSum;
	}

	bitShift(count, complexArr);
	int round1Iterations = log2l(count);

	processLocal(round1Iterations, count, complexArr);
	end = MPI_Wtime();

	printResult(count, complexArr);
	printf("Time used %f\n", end-start);

}



int main(int argc, char **argv) {
	MPI_Init(NULL, NULL);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm split_comm;

	// Split the world into two communication groups for the first half
	// representing
	// nodes that will receive even values, and a second half for the nodes that
	// will receive odd values.
	// For this reason this program will not work when p = 1, despite the fact
	// that  2^0 = 1
	
	int colour = getColour(world_rank, world_size);

	// Split communications based upon colour.
	// This is only used for distributing the initial data.
	MPI_Comm_split(MPI_COMM_WORLD, colour, world_rank, &split_comm);
	int section_rank;
	int section_size;

	MPI_Comm_size(split_comm, &section_size);
	MPI_Comm_rank(split_comm, &section_rank);

	// Count of all varibles
	ULONG count;

	// How many local values do we have
	ULONG localValueCount;

	// The values for the node to compute
	double complex *localValues;
	double complex *remoteValues;

	// This half of the code extracts, then distributes the data
	// to the appropriate half.
	// Once receive in the appropriate half, the bit 
	// shifting operation happens.
	// As this only has a linear complexity, this is only calculated 
	// on the two root nodes for the odd and even communicator groups,
	// rather than incurring additional overhead
	// with messages. The execution time of the bitshift
	// function is only 1 second with
	// 2^25 data points when executed in serial.
	// However there are a number of 
	// more sophisticated algorithms for bit shifting.
	
	
	if (world_rank == 0) {
		start = MPI_Wtime();		
		// Get the number of values
		scanf("%lu", &count);

		// Ensure input size is a power of 2
		assert(ceil(log2(count)) == floor(log2(count)));

		// Ensure that the number of processes does not equal to 1, and the
		// number of processes is a power of 2.
		assert(ceil(log2(world_size))
				== floor(log2(world_size)));


		localValueCount = count / world_size;
		ULONG arrSize = count / 2;

		if (world_size == 1) {
				// As the sequential algorithm is different to the parallel
				// algorithm, we capture this, and process it diffently.
				processSequential(count);
				MPI_Finalize();
				return 0;
		}
	

		double complex *oddArr = malloc(arrSize * sizeof(double complex));

		// As a rank of 0 is always on the even side, we can assume
		// the inputArray is even.
		double complex *inputArr = malloc(arrSize * sizeof(double complex));
		inputArr = malloc((count/2) * sizeof(double complex));

		localValues = malloc(localValueCount * sizeof(double complex));

		// Broadcast the size of the arrays
		MPI_Bcast(&count, 1,  MPI_LONG, 0, MPI_COMM_WORLD);
		
		// Counter for odd index
		ULONG oddIndex = 0;
		ULONG evenIndex = 0;

		double realCompoent;
		double complexCompoent;
		double complex complexSum;
		for (ULONG i = 0; i < count; i++) {
			scanf("%lf %lf", &realCompoent, &complexCompoent);
			complexSum = realCompoent + (complexCompoent * I);
			if (i % 2 == 1) {
				// Add to odd array
				oddArr[oddIndex] = complexSum;
				oddIndex++;
			} else {
				// Add to Even array
				inputArr[evenIndex] = complexSum;
				evenIndex++;
			}
		}

		// Send the Odd array to be bitshifted by the first odd node.
		MPI_Send(oddArr,
			 oddIndex,
			 MPI_C_DOUBLE_COMPLEX,
			 world_size/2,
			 0,
 			 MPI_COMM_WORLD);

		// We don't need the odd array anymore.
		free(oddArr);

		bitShift(arrSize, inputArr);
		
		// Scatter the even value array to all members
		// of the even group
		MPI_Scatter(inputArr,
			    localValueCount,
			    MPI_C_DOUBLE_COMPLEX,
			    localValues,
			    localValueCount,
			    MPI_C_DOUBLE_COMPLEX,
			    0,
			    split_comm);

		free(inputArr);
	} else {
		// Receive the count
		MPI_Bcast(&count, 1,  MPI_LONG, 0, MPI_COMM_WORLD);

		localValueCount = count / world_size;
		localValues = malloc(sizeof(double complex) * localValueCount);

		// This node is receiving the odd values, then performing the bit
		// shift
		if (world_rank == world_size / 2) {
			// Array of all complex numbers for this nodes half.
			double complex *inputArr =
				malloc((count/2) * sizeof(double complex));
			MPI_Recv(inputArr,
				 count/2,
				 MPI_C_DOUBLE_COMPLEX,
				 0,
				 0,
				 MPI_COMM_WORLD,
				 MPI_STATUS_IGNORE);

			// Now perform bit shifting of the odd array.
			bitShift(count/2, inputArr);

			// Scatter the odd numbers
			MPI_Scatter(inputArr,
				    localValueCount,
				    MPI_C_DOUBLE_COMPLEX,
				    localValues,
			            localValueCount,
				    MPI_C_DOUBLE_COMPLEX,
				    0,
				    split_comm);
		}

		if (colour == LOWER && world_rank != world_size / 2) {
			// Receive the Odd values
			MPI_Scatter(NULL,
				    0,
			            MPI_C_DOUBLE_COMPLEX,
				    localValues,
				    localValueCount,
				    MPI_C_DOUBLE_COMPLEX,
				    0,
				    split_comm);
		}

		if (colour == UPPER && world_rank != 0) { 
			// Receive the Even values
			MPI_Scatter(NULL,
				    0,
				    MPI_C_DOUBLE_COMPLEX,
				    localValues,
				    localValueCount,
				    MPI_C_DOUBLE_COMPLEX,
				    0,
				    split_comm);
		}
	}

	// The distribution of the data is complete. Lets calculate the FFT!
	
	// Round 1. Fight!
	// In this round all the local butterflies are calculated.
	// In the binary exchange algorithm, there are log(n/p) stages
	// which can be performed locally without any message
	// passing
	// The idea of this round, is to maximize the computations
	// that are done locally, so that the overhead of messaging
	// is avoided.
	ULONG stepCount = 0;
	ULONG round1Iterations = log2l(count / (ULONG) world_size);

	processLocal(round1Iterations, localValueCount, localValues);

	// The number of local values is always equal to the number of remove
	// values that we receive, due to symmetry
	remoteValues = malloc(sizeof(double complex) * localValueCount);	

	// Now that round 1 is over, we move into round 2.
	// In this round, all butterflies must be solved through the use
	// of communications.
	ULONG round2iterations = log2(world_size);
	for (ULONG r2count = 0; r2count < round2iterations; r2count++) {
		// Figure out if the node is above or below the mid point
		ULONG butterflySize = powl(2, round1Iterations + r2count + 1);


	    	if ((localValueCount * world_rank) % butterflySize >= butterflySize / 2) {
				// In the bottom half of the butterfly.
			
				// Calculate the k value for the twiddle factor
			 	ULONG twiddle = (localValueCount * world_rank) % (butterflySize/2);

				// Apply Twiddle factor to my values
				for (ULONG locCount = 0; locCount < localValueCount; locCount++) {
					localValues[locCount] =
						cexp((-I * (2*M_PI) * twiddle) / butterflySize)
						* localValues[locCount];			
				twiddle++;
			}


			// Compute the node that I am sending to 
			int sendToNode =
				(localValueCount * world_rank - (butterflySize/2))
				/ localValueCount;


			// Send to other node
			MPI_Send(localValues,
				  localValueCount,
				  MPI_C_DOUBLE_COMPLEX,
				  sendToNode,
				  0,
				  MPI_COMM_WORLD);

			
			// Receive Values from other node
			MPI_Recv(remoteValues,
				 localValueCount,
				 MPI_C_DOUBLE_COMPLEX,
				 sendToNode,
				 0,
				 MPI_COMM_WORLD,
				 MPI_STATUS_IGNORE);

			// Simple loop with no side effects for autovectoriser
			for (ULONG locCount = 0; locCount < localValueCount; locCount++) {
				// My value is equal to the values received, plus my twiddle
				localValues[locCount] =
					remoteValues[locCount] - localValues[locCount];
			}
			
		} else {
			// Receive values with Twiddle factors applied from other node			
			int sendToNode = 
				(localValueCount * world_rank + (butterflySize/2))
				/ localValueCount;

			MPI_Recv(remoteValues,
				 localValueCount,
				 MPI_C_DOUBLE_COMPLEX,
				 sendToNode,
				 0,
				 MPI_COMM_WORLD,
				 MPI_STATUS_IGNORE);

			MPI_Send(localValues,
				 localValueCount,
				 MPI_C_DOUBLE_COMPLEX,
				 sendToNode,
				 0,
				 MPI_COMM_WORLD);

			for (ULONG locCount = 0; locCount < localValueCount; locCount++) {
				localValues[locCount]
					= localValues[locCount] + remoteValues[locCount];
			}

		}
	}

	// No Longer need to store remote values
	free(remoteValues);

	double complex *allValues = NULL;
	ULONG receiveCount = 0;
	// Now gather the values
	if (world_rank == 0) {
		receiveCount = localValueCount;
		allValues = malloc(sizeof(double complex) * (world_size * localValueCount));	
	}

	MPI_Gather(localValues,
		   localValueCount,
		   MPI_C_DOUBLE_COMPLEX,
		   allValues,
		   receiveCount,
		   MPI_C_DOUBLE_COMPLEX,
		   0,
		   MPI_COMM_WORLD);


	if (world_rank == 0) {
		end = MPI_Wtime();		
		printResult(world_size * localValueCount, allValues);
		printf("Time Consumed %f\n", end-start);
	}
	
	free(localValues);
	free(allValues);
	MPI_Finalize();	
	return 0;
}
