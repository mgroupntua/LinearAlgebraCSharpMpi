using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Text;
using MPI;

namespace LinAlgMpi.LinearAlgebra
{
    public static class IterativeMethodsMpi
    {
		public static void SolveJacobi(Intracommunicator comm, int n, double[,] A, double[] b, double[] x, int maxIterations, double tolerance)
		{
			double[] invD = MpiBLAS.InvertDiagonalStriped(comm, n, A);

			// x(t+1) = x(t) - inv(D) * A * x(t) + inv(D) * b 
			double[] c = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x);
			MpiBLAS.MultiplyPointwise(comm, n, invD, b, c);
			for (int t = 0; t < maxIterations; t++)
			{
				double[] w = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x); // dummy vector
				double[] xNew = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x); // dummy vector
				MpiBLAS.MultiplyMatrixVectorStriped(comm, n, n, A, x, w); // w = A * x
				MpiBLAS.MultiplyPointwise(comm, n, invD, w, w); // w = invD * (A*x)
				MpiBLAS.Axpby(comm, n, +1, x, -1, w, w); // w = x - (D*A*x)
				MpiBLAS.Axpby(comm, n, +1, w, +1, c, xNew); // xNew = (x - D*A*x) + c

				MpiBLAS.Axpby(comm, n, +1, xNew, -1, x, w); // w = x(t+1) - x(t)
				double error = Math.Sqrt(MpiBLAS.DotProduct(comm, n, w, w)); // ||x(t+1) - x(t)||
                DistributedUtilities.CopyDistributedVector(comm, xNew, x);
				if (error < tolerance) return;
			}
		}

		public static void SolvePcg(Intracommunicator comm, int n, double[,] A, double[] b, double[] x, int maxIterations, double tolerance)
		{
			// Create preconditioner
			double[] diagM = MpiBLAS.InvertDiagonalStriped(comm, n, A);

			// Initialize quantities used in PCG
			double[] r = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x); // residual
			double[] p = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x); // direction vector
			double[] q = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x); // matrix * direction vector
			double[] z = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x); // perconditioner * residual
			double zr = double.NaN; // z * r
			double zrSqrt0 = double.NaN; // sqrt(z * r) of the initial iteration

			// Initial iteration
			// r = b - A*x
			MpiBLAS.MultiplyMatrixVectorStriped(comm, n, n, A, x, r);
			MpiBLAS.Axpby(comm, n, +1, b, -1, r, r);

			// z = M * r
			MpiBLAS.MultiplyPointwise(comm, n, diagM, r, z);

			// z * r
			zr = MpiBLAS.DotProduct(comm, n, z, r);
			zrSqrt0 = Math.Sqrt(zr);

            // p = z
            DistributedUtilities.CopyDistributedVector(comm, z, p);

			// q = A * p
			MpiBLAS.MultiplyMatrixVectorStriped(comm, n, n, A, p, q);

			// alpha = z*r / p*q
			double alpha = MpiBLAS.DotProduct(comm, n, z, r) / MpiBLAS.DotProduct(comm, n, p, q);

			for (int t = 0; t < maxIterations; t++)
			{
				// x = x + alpha * p
				MpiBLAS.Axpby(comm, n, +1, x, alpha, p, x);

				// r = r - alpha * q
				MpiBLAS.Axpby(comm, n, +1, r, -alpha, q, r);

				// z = M * r
				MpiBLAS.MultiplyPointwise(comm, n, diagM, r, z);

				// if sqrt(z(t+1)*r(t+1)) / sqrt(z(0)*r(0)) < tolerance, then PCG has converged 
				double zrNext = MpiBLAS.DotProduct(comm, n, z, r);
				Debug.WriteLine(Math.Sqrt(zrNext) / zrSqrt0);
				if (Math.Sqrt(zrNext) / zrSqrt0 < tolerance) return;

				// beta = z(t+1)*r(t+1) / z(t)*r(t)
				double beta = zrNext / zr;
				zr = zrNext;

				// p = z + beta * p
				MpiBLAS.Axpby(comm, n, +1, z, beta, p, p);

				// q = A * p
				MpiBLAS.MultiplyMatrixVectorStriped(comm, n, n, A, p, q);

				// alpha = z*r / p*q
				alpha = MpiBLAS.DotProduct(comm, n, z, r) / MpiBLAS.DotProduct(comm, n, p, q);
			}
		}
	}
}
