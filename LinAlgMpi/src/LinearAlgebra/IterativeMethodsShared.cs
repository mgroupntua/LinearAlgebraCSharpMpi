using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Text;
using MPI;

namespace LinAlgMPI.LinearAlgebra
{
    public static class IterativeMethodsShared
    {
        public static void SolveJacobi(double[,] A, double[] b, double[] x, int maxIterations, double tolerance)
        {
            int n = b.Length;
            double[] invD = SharedBLAS.InvertDiagonal(n, A);

            // x(t+1) = x(t) - inv(D) * A * x(t) + inv(D) * b 
            double[] c = new double[n];
            SharedBLAS.MultiplyPointwise(n, invD, b, c);
            for (int t = 0; t < maxIterations; t++)
            {
                double[] w = new double[n]; // dummy vector
                double[] xNew = new double[n]; // dummy vector
                SharedBLAS.MultiplyMatrixVectorStriped(n, n, A, x, w); // w = A * x
                SharedBLAS.MultiplyPointwise(n, invD, w, w); // w = invD * (A*x)
                SharedBLAS.Axpby(n, +1, x, -1, w, w); // w = x - (D*A*x)
                SharedBLAS.Axpby(n, +1, w, +1, c, xNew); // xNew = (x - D*A*x) + c

                SharedBLAS.Axpby(n, +1, xNew, -1, x, w); // w = x(t+1) - x(t)
                double error = Math.Sqrt(SharedBLAS.DotProduct(n, w, w)); // ||x(t+1) - x(t)||
                Array.Copy(xNew, x, n); // x = xNew
                if (error < tolerance) return;
            }
        }

        public static void SolvePCG(double[,] A, double[] b, double[] x, int maxIterations, double tolerance)
        {
            int n = b.Length;

            // Create preconditioner
            double[] diagM = SharedBLAS.InvertDiagonal(n, A);

            // Initialize quantities used in PCG
            double[] r = new double[n]; // residual
            double[] p = new double[n]; // direction vector
            double[] q = new double[n]; // matrix * direction vector
            double[] z = new double[n]; // perconditioner * residual
            double zr = double.NaN; // z * r
            double zrSqrt0 = double.NaN; // sqrt(z * r) of the initial iteration

            // Initial iteration

            // r = b - A*x
            SharedBLAS.MultiplyMatrixVectorStriped(n, n, A, x, r);
            SharedBLAS.Axpby(n, +1, b, -1, r, r);

            // z = M * r
            SharedBLAS.MultiplyPointwise(n, diagM, r, z);

            // z * r
            zr = SharedBLAS.DotProduct(n, z, r);
            zrSqrt0 = Math.Sqrt(zr);

            // p = z
            Array.Copy(z, p, n);

            // q = A * p
            SharedBLAS.MultiplyMatrixVectorStriped(n, n, A, p, q);

            // alpha = z*r / p*q
            double alpha = SharedBLAS.DotProduct(n, z, r) / SharedBLAS.DotProduct(n, p, q);

            for (int t = 0; t < maxIterations; t++)
            {
                // x = x + alpha * p
                SharedBLAS.Axpby(n, +1, x, alpha, p, x);

                // r = r - alpha * q
                SharedBLAS.Axpby(n, +1, r, -alpha, q, r);

                // z = M * r
                SharedBLAS.MultiplyPointwise(n, diagM, r, z);

                // if sqrt(z(t+1)*r(t+1)) / sqrt(z(0)*r(0)) < tolerance, then PCG has converged 
                double zrNext = SharedBLAS.DotProduct(n, z, r);
                Debug.WriteLine(Math.Sqrt(zrNext) / zrSqrt0);
                if (Math.Sqrt(zrNext) / zrSqrt0 < tolerance) return;

                // beta = z(t+1)*r(t+1) / z(t)*r(t)
                double beta = zrNext / zr;
                zr = zrNext;

                // p = z + beta * p
                SharedBLAS.Axpby(n, +1, z, beta, p, p);

                // q = A * p
                SharedBLAS.MultiplyMatrixVectorStriped(n, n, A, p, q);

                // alpha = z*r / p*q
                alpha = SharedBLAS.DotProduct(n, z, r) / SharedBLAS.DotProduct(n, p, q);
            }
        }
    }
}
