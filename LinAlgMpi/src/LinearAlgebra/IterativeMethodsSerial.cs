using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Text;
using MPI;

namespace LinAlgMPI.LinearAlgebra
{
    public static class IterativeMethodsSerial
    {
        public static void SolveJacobi(double[,] A, double[] b, double[] x, int maxIterations, double tolerance)
        {
            int n = b.Length;
            double[] invD = SerialBLAS.InvertDiagonal(n, A);

            // x(t+1) = x(t) - inv(D) * A * x(t) + inv(D) * b 
            double[] c = new double[n];
            SerialBLAS.MultiplyPointwise(n, invD, b, c);
            for (int t = 0; t < maxIterations; t++)
            {
                double[] w = new double[n]; // dummy vector
                double[] xNew = new double[n]; // dummy vector
                SerialBLAS.MultiplyMatrixVector(n, n, A, x, w); // w = A * x
                SerialBLAS.MultiplyPointwise(n, invD, w, w); // w = invD * (A*x)
                SerialBLAS.Axpby(n, +1, x, -1, w, w); // w = x - (D*A*x)
                SerialBLAS.Axpby(n, +1, w, +1, c, xNew); // xNew = (x - D*A*x) + c

                SerialBLAS.Axpby(n, +1, xNew, -1, x, w); // w = x(t+1) - x(t)
                double error = Math.Sqrt(SerialBLAS.DotProduct(n, w, w)); // ||x(t+1) - x(t)||
                Array.Copy(xNew, x, n); // x = xNew
                if (error < tolerance) return;
            }
        }

        public static void SolvePCG(double[,] A, double[] b, double[] x, int maxIterations, double tolerance)
        {
            int n = b.Length;

            // Create preconditioner
            double[] diagM = SerialBLAS.InvertDiagonal(n, A);

            // Initialize quantities used in PCG
            double[] r = new double[n]; // residual
            double[] p = new double[n]; // direction vector
            double[] q = new double[n]; // matrix * direction vector
            double[] z = new double[n]; // perconditioner * residual
            double zr = double.NaN; // z * r
            double zrSqrt0 = double.NaN; // sqrt(z * r) of the initial iteration

            // Initial iteration

            // r = b - A*x
            SerialBLAS.MultiplyMatrixVector(n, n, A, x, r);
            SerialBLAS.Axpby(n, +1, b, -1, r, r);

            // z = M * r
            SerialBLAS.MultiplyPointwise(n, diagM, r, z);

            // z * r
            zr = SerialBLAS.DotProduct(n, z, r);
            zrSqrt0 = Math.Sqrt(zr);

            // p = z
            Array.Copy(z, p, n);

            // q = A * p
            SerialBLAS.MultiplyMatrixVector(n, n, A, p, q);

            // alpha = z*r / p*q
            double alpha = SerialBLAS.DotProduct(n, z, r) / SerialBLAS.DotProduct(n, p, q);
                
            for (int t = 0; t < maxIterations; t++)
            {
                // x = x + alpha * p
                SerialBLAS.Axpby(n, +1, x, alpha, p, x);

                // r = r - alpha * q
                SerialBLAS.Axpby(n, +1, r, -alpha, q, r);

                // z = M * r
                SerialBLAS.MultiplyPointwise(n, diagM, r, z);

                // if sqrt(z(t+1)*r(t+1)) / sqrt(z(0)*r(0)) < tolerance, then PCG has converged 
                double zrNext = SerialBLAS.DotProduct(n, z, r);
                Debug.WriteLine(Math.Sqrt(zrNext) / zrSqrt0);
                if (Math.Sqrt(zrNext) / zrSqrt0 < tolerance) return;

                // beta = z(t+1)*r(t+1) / z(t)*r(t)
                double beta = zrNext / zr;
                zr = zrNext;

                // p = z + beta * p
                SerialBLAS.Axpby(n, +1, z, beta, p, p);

                // q = A * p
                SerialBLAS.MultiplyMatrixVector(n, n, A, p, q);

                // alpha = z*r / p*q
                alpha = SerialBLAS.DotProduct(n, z, r) / SerialBLAS.DotProduct(n, p, q);
            }
        }
    }
}
