using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MPI;

namespace SeminarMpi.LinearAlgebra
{
	public class PcgSolver
	{
		public static void SolveSerial(int n, double[] A, double[] b, double[] x, int maxIterations, double tolerance)
		{
			// Create preconditioner
			double[] diagM = SerialBLAS.InvertDiagonal(n, A);

			// Initialize quantities used in PCG
			double[] r = new double[b.Length]; // residual
			double[] p = new double[b.Length]; // direction vector
			double[] q = new double[b.Length]; // matrix * direction vector
			double[] z = new double[b.Length]; // perconditioner * residual
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

		public static void SolveTpl(int n, double[] A, double[] b, double[] x, int maxIterations, double tolerance)
		{
			// Create preconditioner
			double[] diagM = TplBLAS.InvertDiagonal(n, A);

			// Initialize quantities used in PCG
			double[] r = new double[b.Length]; // residual
			double[] p = new double[b.Length]; // direction vector
			double[] q = new double[b.Length]; // matrix * direction vector
			double[] z = new double[b.Length]; // perconditioner * residual
			double zr = double.NaN; // z * r
			double zrSqrt0 = double.NaN; // sqrt(z * r) of the initial iteration

			// Initial iteration

			// r = b - A*x
			TplBLAS.MultiplyMatrixVector(n, n, A, x, r);
			TplBLAS.Axpby(n, +1, b, -1, r, r);

			// z = M * r
			TplBLAS.MultiplyPointwise(n, diagM, r, z);

			// z * r
			zr = TplBLAS.DotProduct(n, z, r);
			zrSqrt0 = Math.Sqrt(zr);

			// p = z
			Array.Copy(z, p, n);

			// q = A * p
			TplBLAS.MultiplyMatrixVector(n, n, A, p, q);

			// alpha = z*r / p*q
			double alpha = TplBLAS.DotProduct(n, z, r) / TplBLAS.DotProduct(n, p, q);

			for (int t = 0; t < maxIterations; t++)
			{
				// x = x + alpha * p
				TplBLAS.Axpby(n, +1, x, alpha, p, x);

				// r = r - alpha * q
				TplBLAS.Axpby(n, +1, r, -alpha, q, r);

				// z = M * r
				TplBLAS.MultiplyPointwise(n, diagM, r, z);

				// if sqrt(z(t+1)*r(t+1)) / sqrt(z(0)*r(0)) < tolerance, then PCG has converged 
				double zrNext = TplBLAS.DotProduct(n, z, r);
				Debug.WriteLine(Math.Sqrt(zrNext) / zrSqrt0);
				if (Math.Sqrt(zrNext) / zrSqrt0 < tolerance) return;

				// beta = z(t+1)*r(t+1) / z(t)*r(t)
				double beta = zrNext / zr;
				zr = zrNext;

				// p = z + beta * p
				TplBLAS.Axpby(n, +1, z, beta, p, p);

				// q = A * p
				TplBLAS.MultiplyMatrixVector(n, n, A, p, q);

				// alpha = z*r / p*q
				alpha = TplBLAS.DotProduct(n, z, r) / TplBLAS.DotProduct(n, p, q);
			}
		}

		public static void SolveMpi(Intracommunicator comm, int n, double[] A, double[] b, double[] x, int maxIterations,
			double tolerance)
		{
			// Create preconditioner
			double[] diagM = MpiBLAS.InvertDiagonal(comm, n, A);

			// Initialize quantities used in PCG
			double[] r = new double[b.Length]; // residual
			double[] p = new double[b.Length]; // direction vector
			double[] q = new double[b.Length]; // matrix * direction vector
			double[] z = new double[b.Length]; // perconditioner * residual
			double zr = double.NaN; // z * r
			double zrSqrt0 = double.NaN; // sqrt(z * r) of the initial iteration

			// Initial iteration

			// r = b - A*x
			MpiBLAS.MultiplyMatrixVector(comm, n, n, A, x, r);
			MpiBLAS.Axpby(comm, n, +1, b, -1, r, r);

			// z = M * r
			MpiBLAS.MultiplyPointwise(comm, n, diagM, r, z);

			// z * r
			zr = MpiBLAS.DotProduct(comm, n, z, r);
			zrSqrt0 = Math.Sqrt(zr);

			// p = z
			Array.Copy(z, p, z.Length);

			// q = A * p
			MpiBLAS.MultiplyMatrixVector(comm, n, n, A, p, q);

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
				MpiBLAS.MultiplyMatrixVector(comm, n, n, A, p, q);

				// alpha = z*r / p*q
				alpha = MpiBLAS.DotProduct(comm, n, z, r) / MpiBLAS.DotProduct(comm, n, p, q);
			}
		}
	}
}
