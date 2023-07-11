using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MPI;

namespace SeminarMpi.LinearAlgebra
{
    public static class SerialBLAS
    {
        public static void Axpby(int n, double a, double[] x, double b, double[] y, double[] result)
        {
            for (int i = 0; i < n; i++)
            {
                result[i] = a * x[i] + b * y[i];
            }
        }

        public static double DotProduct(int n, double[] x, double[] y)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
            {
                sum += x[i] * y[i];
            }
            return sum;
        }

        public static double[] InvertDiagonal(int n, double[] A)
        {
            double[] invD = new double[n];
            for (int i = 0; i < n; i++)
            {
                int t = i * n + i;
                invD[i] = 1.0 / A[t];
            }
            return invD;
        }

        public static void MultiplyMatrixVector(int m, int n, double[] A, double[] x, double[] b)
        {
            for (int i = 0; i < m; i++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    int t = i * n + j;
                    sum += A[t] * x[j];
                }
                b[i] = sum;
            }
        }

        public static void MultiplyPointwise(int n, double[] x, double[] y, double[] result)
        {
            for (int i = 0; i < n; i++)
            {
                result[i] = x[i] * y[i];
            }
        }
    }
}
