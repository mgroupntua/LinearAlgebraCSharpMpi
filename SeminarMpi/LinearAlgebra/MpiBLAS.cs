using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;
using MPI;
using SeminarMpi.Utilities;

namespace SeminarMpi.LinearAlgebra
{
    public static class MpiBLAS
    {
        public static void Axpby(Intracommunicator comm, int n, double a, double[] x, double b, double[] y, double[] result)
        {
            int ns = x.Length;
            Debug.Assert(y.Length == ns);
            Debug.Assert(result.Length == ns);

            for (int i = 0; i < ns; i++)
            {
                result[i] = a * x[i] + b * y[i];
            }
        }

        public static double[] CreateZeroVector(Intracommunicator comm, int n)
        {
            int[] chunkSizes = DataTransfers.FindChunkSizes(comm.Size, n);
            int ns = chunkSizes[comm.Rank];
			return new double[ns];
        }

        public static double DotProduct(Intracommunicator comm, int n, double[] x, double[] y)
        {
            int ns = x.Length;
            Debug.Assert(y.Length == ns);

            // Calculate dot product of this process's subvector.
            double partialSum = 0;
            for (int i = 0; i < ns; i++)
            {
                partialSum += x[i] * y[i];
            }

            // Sum partial results between different processes. The total result will only be available in root process
            int root = 0;
            double totalSum = comm.Reduce(partialSum, Operation<double>.Add, Constants.masterProcess);

            // Send the total result to all other processes
            comm.Broadcast(ref totalSum, root);
            return totalSum;
        }

        public static double[] InvertDiagonal(Intracommunicator comm, int n, double[] A)
        {
            int ms = A.Length / n;
            Debug.Assert(A.Length % n == 0);
            double[] invD = new double[ms];
            
            //Find the first row of the global matrix for this process
            int[] numRowsPerProcess = DataTransfers.FindChunkSizes(comm.Size, n);
            int firstRow = 0;
            for (int r = 0; r < comm.Rank; r++)
            {
                firstRow += numRowsPerProcess[r];
            }

			for (int i = 0; i < ms; i++)
            {
                int I = firstRow + i;
                int t = i * n + I;
				invD[i] = 1.0 / A[t];
            }
            return invD;
        }

        public static void MultiplyMatrixVector(Intracommunicator comm, int m, int n,
            double[] A, double[] x, double[] b)
        {
            // All processes need access to the whole x vector, in order to multiply rows of the matrix with it
            double[] xGlob = DataTransfers.GatherVector(comm, n, x);
            if (comm.Rank == Constants.masterProcess)
            {
				Debug.Assert(xGlob.Length == n);
			}
			else 
            {
                xGlob = new double[n]; // Otherwise, it would be null in all processes other than master
            }
            comm.Broadcast(ref xGlob, Constants.masterProcess);

            // In contrast, each process only owns a part of the total result vector b. 
            int ms = b.Length;
            Debug.Assert(ms * n == A.Length);

            // Calculate the relevant entries of the result vector b.
            for (int i = 0; i < ms; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    sum += A[i * n + j] * xGlob[j];
                }
                b[i] = sum;
            }
        }

        public static void MultiplyPointwise(Intracommunicator comm, int n, double[] x, double[] y, double[] result)
        {
            int ns = x.Length;
            Debug.Assert(y.Length == ns);
            Debug.Assert(result.Length == ns);

            for (int i = 0; i < ns; i++)
            {
                result[i] = x[i] * y[i];
            }
        }
    }
}
