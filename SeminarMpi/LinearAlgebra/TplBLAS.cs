using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MPI;

namespace SeminarMpi.LinearAlgebra
{
    public static class TplBLAS
    {

        public static void Axpby(int n, double a, double[] x, double b, double[] y, double[] result)
        {
            int numThreads = System.Environment.ProcessorCount;
            int ns = (n - 1) / numThreads + 1; // CEILING(numEntries / numThreads)

            Parallel.For(0, numThreads, (p) =>
            {
                int start = ns * p;
                int end = Math.Min(start + ns, n); // exclusive
                for (int i = start; i < end; i++) // The values of i for one thread do not overlap with another. 
                {
                    result[i] = a * x[i] + b * y[i]; // No overlap in i, thus no race condition
                }
            });
        }

        public static double DotProduct(int n, double[] x, double[] y)
        {
            int numThreads = System.Environment.ProcessorCount;
            int ns = (n - 1) / numThreads + 1; // CEILING(numEntries / numThreads)

            // Calculate dot products of subvectors
            double[] partialSums = new double[numThreads];
            Parallel.For(0, numThreads, (p) =>
            {
                int start = ns * p;
                int end = Math.Min(start + ns, n); // exclusive
                for (int i = start; i < end; i++)
                {
                    partialSums[p] += x[i] * y[i];
                }
            });

            // Sum partial results serially
            double totalSum = 0;
            for (int p = 0; p < numThreads; ++p)
            {
                totalSum += partialSums[p];
            }

            return totalSum;
        }

        public static double[] InvertDiagonal(int n, double[] A)
        {
            int numThreads = System.Environment.ProcessorCount;
            int ms = (n - 1) / numThreads + 1; // CEILING(numRows / numThreads)

            double[] invD = new double[n];
            Parallel.For(0, numThreads, (p) =>
            {
                int start = ms * p;
                int end = Math.Min(start + ms, n); // exclusive
                for (int i = start; i < end; i++) // The values of i for one thread do not overlap with another. 
                {
                    int t = i * n + i;
                    invD[i] = 1.0 / A[t]; // No overlap in i, thus no race condition
                }
            });

            return invD;
        }

        public static void MultiplyMatrixVector(int m, int n, double[] A, double[] x, double[] b)
        {
            int numThreads = System.Environment.ProcessorCount;
            int ms = (m - 1) / numThreads + 1; // CEILING(numRows / numThreads)

            // Each thread operates on a subset of matrix rows and the corresponding entries of the rhs vector
            Parallel.For(0, numThreads, (p) =>
            {
                int startRow = ms * p;
                int endRow = Math.Min(startRow + ms, m); // exclusive
                for (int i = startRow; i < endRow; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < n; j++)
                    {
                        int t = i * n + j;
                        sum += A[t] * x[j];
                    }
                    b[i] = sum;
                }
            });
        }

        public static void MultiplyPointwise(int n, double[] x, double[] y, double[] result)
        {
            int numThreads = System.Environment.ProcessorCount;
            int ns = (n - 1) / numThreads + 1; // CEILING(numEntries / numThreads)

            Parallel.For(0, numThreads, (p) =>
            {
                int start = ns * p;
                int end = Math.Min(start + ns, n); // exclusive
                for (int i = start; i < end; i++) // The values of i for one thread do not overlap with another. 
                {
                    result[i] = x[i] * y[i]; // No overlap in i, thus no race condition
                }
            });
        }
    }
}
