using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MPI;

namespace LinAlgMPI.LinearAlgebra
{
    public static class MpiBLAS
    {
        public static void AxpbyMirror(Intracommunicator comm, int n, double a, double[] x, double b, double[] y, 
            double[] result)
        {
            // We need the result in all processes. Therefore we have to do all operations for each process.
            for (int i = 0; i < n; i++)
            {
                result[i] = a * x[i] + b * y[i];
            }
        }

        public static void AxpbyDistributed(Intracommunicator comm, int n, double a, double[] x, double b, double[] y, 
            double[] result)
        {
            // Calculate dot product of this process's subvector. Avoid padding entries
            int numProcesses = comm.Size;
            int chunkSize = (n - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)
            int startGlobal = comm.Rank * chunkSize;
            int endLocal = Math.Min(chunkSize, n - startGlobal);
            for (int i = 0; i < endLocal; i++)
            {
                result[i] = a * x[i] + b * y[i];
            }
        }
        
        public static double DotProductMirror(Intracommunicator comm, int n, double[] x, double[] y)
        {
            int numProcesses = comm.Size;
            int chunkSize = (n - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

            // Calculate dot product of this process's subvector, by accessing only the relevant entries
            int start = chunkSize * comm.Rank;
            int end = Math.Min(start + chunkSize, n); // exclusive
            double partialSum = 0;
            for (int i = start; i < end; i++)
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

        public static double DotProductDistributed(Intracommunicator comm, int n, double[] x, double[] y)
        {
            // Calculate dot product of this process's subvector. Avoid padding entries
            int numProcesses = comm.Size;
            int chunkSize = (n - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)
            int startGlobal = comm.Rank * chunkSize;
            int endLocal = Math.Min(chunkSize, n - startGlobal);
            double partialSum = 0;
            for (int i = 0; i < endLocal; i++)
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
            double[] invDLocal = new double[endRow - startRow];
            for (int i = 0; i < endRow - startRow; i++)
            {
                int iGlobal = startRow + i;
                invDLocal[i] = 1.0 / A[i, iGlobal];
            }

            double[] invD = new double[n];
            DistributedToMirrorVector(comm, invD, invDLocal);

            return invD;
        }

        public static void MultiplyMatrixVectorMirrorStriped(Intracommunicator comm, int m, int n, 
            double[,] A, double[] x, double[] b)
        {
            Array.Clear(b, 0, n);
            int numProcesses = comm.Size;
            int chunkSize = (n - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

            // Calculate local parts of the total result vector
            double[] bLocal = new double[chunkSize];
            int startRow = chunkSize * comm.Rank;
            int endRow = Math.Min(startRow + chunkSize, n); // exclusive
            for (int i = 0; i < endRow - startRow; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    bLocal[i] += A[i, j] * x[j];
                }
            }

            // Right now the local vector of each process only stores a part of the total result.
            // In each process we need a global vector that contains all entries
            DistributedToMirrorVector(comm, b, bLocal);
        }

        public static void MultiplyPointwiseMirror(Intracommunicator comm, int n, double[] x, double[] y, double[] result)
        {
            // We need the result in all processes. Therefore we have to do all operations for each process.
            for (int i = 0; i < n; i++)
            {
                result[i] = x[i] * y[i];
            }
        }

        public static void DistributedToMirrorVector(Intracommunicator comm, double[] globalVector, double[] localVector)
        {
            int globalSize = globalVector.Length;
            int numProcesses = comm.Size;
            int chunkSize = (globalSize - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

            // Gather all local vectors to all processes
            double[][] localVectors = new double[numProcesses][];
            comm.Allgather(localVector, ref localVectors); // More efficient than the next commented-out code
            //int root = 0;
            //comm.Gather(localVector, root, ref localVectors);
            //comm.Broadcast(ref localVectors, root);

            // Each process copies all these local vectors into the global vector
            for (int p = 0; p < numProcesses; p++)
            {
                int startY = p * chunkSize;
                int count = Math.Min(chunkSize, globalSize - startY);
                Array.Copy(localVectors[p], 0, globalVector, startY, count);
                startY += chunkSize;
            }
        }
    }
}
