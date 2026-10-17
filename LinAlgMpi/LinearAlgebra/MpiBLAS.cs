using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MPI;

namespace LinAlgMpi.LinearAlgebra
{
    public static class MpiBLAS
    {
		public static void Axpby(Intracommunicator comm, int n, double a, double[] x, double b, double[] y, double[] result)
		{
			// Each process gets an array with only its corresponding entries
			for (int i = 0; i < x.Length; i++)
			{
				result[i] = a * x[i] + b * y[i];
			}
		}

		public static double DotProduct(Intracommunicator comm, int n, double[] x, double[] y)
		{
			// Calculate dot product of this process's subvector
            double partialSum = 0;
			for (int i = 0; i < x.Length; i++)
			{
				partialSum += x[i] * y[i];
			}

			// Sum partial results between different processes. The total result will only be available in root process
			int root = 0;
			double totalSum = comm.Reduce(partialSum, Operation<double>.Add, root);

			// Send the total result to all other processes
			comm.Broadcast(ref totalSum, root);
			return totalSum;
		}

		public static double[] InvertDiagonalStriped(Intracommunicator comm, int n, double[,] A)
		{
			int numProcesses = comm.Size;
			int chunkSize = (n - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)
			int startRow = chunkSize * comm.Rank;
			int endRow = Math.Min(startRow + chunkSize, n); // exclusive

			double[] invD = new double[endRow - startRow];
			for (int i = 0; i < endRow - startRow; i++)
			{
				int iGlobal = startRow + i;
				invD[i] = 1.0 / A[i, iGlobal];
			}

			return invD;
		}

		public static void MultiplyMatrixVectorStriped(Intracommunicator comm, int m, int n, double[,] A, double[] x, double[] b)
		{
			// Gather the whole vector x
			var xGlobal = new double[n];
			DistributedUtilities.DistributedToMirrorVector(comm, xGlobal, x);

            // Define the bounds of this thread
			int numProcesses = comm.Size;
			int chunkSize = (n - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)
			int startRow = chunkSize * comm.Rank;
			int endRow = Math.Min(startRow + chunkSize, n); // exclusive

			// Calculate local parts of the total result vector
            Array.Clear(b);
			for (int i = 0; i < endRow - startRow; i++)
			{
				for (int j = 0; j < n; ++j)
				{
					b[i] += A[i, j] * xGlobal[j];
				}
			}
		}

		public static void MultiplyPointwise(Intracommunicator comm, int n, double[] x, double[] y, double[] result)
		{
			// Each process gets an array with only its corresponding entries
			for (int i = 0; i < x.Length; i++)
			{
				result[i] = x[i] * y[i];
			}
		}
    }
}
