using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MPI;
using Xunit;
using LinAlgMpi.LinearAlgebra;
using LinAlgMpi.Tests;

namespace LinAlgMpi.Tests
{
    public class MpiTests
    {
		public static void TestAxpby(Intracommunicator comm)
		{
			double[] x = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorX);
			double[] y = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorY);
			double[] result = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x);

			MpiBLAS.Axpby(comm, TestProblem.Size, 2, x, 3, y, result);
			double[] expected = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.Vector2X3Y);

			Utilities.AssertEqual(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestDotProduct(Intracommunicator comm)
		{
			// Each process will own only a part of the whole vector.
			double[] x = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorX);
			double[] y = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorY);

			double result = MpiBLAS.DotProduct(comm, TestProblem.Size, x, y);
			double expected = TestProblem.dotXY;

			Assert.Equal(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestMultiplyMVStriped(Intracommunicator comm)
		{
			// Each process will own only a part of the whole vector.
			double[,] A = DistributedUtilities.MirroredToStripedMatrix(comm, TestProblem.MatrixA);
			double[] x = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorX);

			int size = TestProblem.Size;
			double[] result = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, x);
			MpiBLAS.MultiplyMatrixVectorStriped(comm, size, size, A, x, result);
			double[] expected = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorAx);

			Utilities.AssertEqual(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestSolveJacobi(Intracommunicator comm)
		{
			double[,] A = DistributedUtilities.MirroredToStripedMatrix(comm, TestProblem.MatrixA);
			double[] b = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorAx);
			double[] expected = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorX);
			double[] x = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, b);
			int n = TestProblem.Size;

			int numIterations = 100;
			double tolerance = 1E-7;
			IterativeMethodsMpi.SolveJacobi(comm, n, A, b, x, numIterations, tolerance);

			Utilities.AssertEqual(expected, x, 3);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestSolvePcg(Intracommunicator comm)
		{
			double[,] A = DistributedUtilities.MirroredToStripedMatrix(comm, TestProblem.MatrixA);
			double[] b = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorAx);
			double[] expected = DistributedUtilities.MirroredToDistributedVector(comm, TestProblem.VectorX);
			double[] x = DistributedUtilities.CreateDistributedVectorWithSameFormat(comm, b);
			int n = TestProblem.Size;

			int numIterations = 100;
			double tolerance = 1E-7;
			IterativeMethodsMpi.SolvePcg(comm, n, A, b, x, numIterations, tolerance);

			Utilities.AssertEqual(expected, x, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}
	}
}
