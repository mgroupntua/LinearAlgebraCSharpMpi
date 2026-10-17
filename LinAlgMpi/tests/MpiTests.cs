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
		public static void TestAxpbyMirror(Intracommunicator comm)
		{
			double[] x = TestProblem.VectorX;
			double[] y = TestProblem.VectorY;
			double[] result = new double[TestProblem.Size];
			MpiBLAS.AxpbyMirror(comm, TestProblem.Size, 2, x, 3, y, result);
			double[] expected = TestProblem.Vector2X3Y;

			Utilities.AssertEqual(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

        public static void TestAxpbyDistributed(Intracommunicator comm)
        {
			double[] x = Utilities.DistributeVector(comm, TestProblem.VectorX);
			double[] y = Utilities.DistributeVector(comm, TestProblem.VectorY);
			double[] result = Utilities.DistributeVector(comm, new double[TestProblem.Size]);

			MpiBLAS.AxpbyDistributed(comm, TestProblem.Size, 2, x, 3, y, result);
			double[] expected = Utilities.DistributeVector(comm, TestProblem.Vector2X3Y);

			Utilities.AssertEqual(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestDotProductMirror(Intracommunicator comm)
		{
			// Each process will own the whole vector, but only operate on a subvector.
			double[] x = TestProblem.VectorX;
			double[] y = TestProblem.VectorY;

			double result = MpiBLAS.DotProductMirror(comm, TestProblem.Size, x, y);
			double expected = TestProblem.dotXY;

			Assert.Equal(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestDotProductDistributed(Intracommunicator comm)
		{
			// Each process will own only a part of the whole vector.
			double[] x = Utilities.DistributeVector(comm, TestProblem.VectorX);
			double[] y = Utilities.DistributeVector(comm, TestProblem.VectorY);

			double result = MpiBLAS.DotProductDistributed(comm, TestProblem.Size, x, y);
			double expected = TestProblem.dotXY;

			Assert.Equal(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestMultiplyMVMirrorStriped(Intracommunicator comm)
        {
			// Each process will own only a part of the whole vector.
			double[,] A = Utilities.DistributeMatrixStriped(comm, TestProblem.MatrixA);
			double[] x = TestProblem.VectorX;

			int size = TestProblem.Size;
			double[] result = new double[size];
			MpiBLAS.MultiplyMatrixVectorMirrorStriped(comm, size, size, A, x, result);
			double[] expected = TestProblem.VectorAx;

			Utilities.AssertEqual(expected, result, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestSolveJacobi(Intracommunicator comm)
		{
			double[,] A = Utilities.DistributeMatrixStriped(comm, TestProblem.MatrixA);
			double[] b = TestProblem.VectorAx;
			double[] expected = TestProblem.VectorX;
			double[] x = new double[TestProblem.Size];

			int numIterations = 100;
			double tolerance = 1E-7;
			IterativeMethodsMpi.SolveJacobi(comm, A, b, x, numIterations, tolerance);

			Utilities.AssertEqual(expected, x, 3);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}

		public static void TestSolvePCG(Intracommunicator comm)
		{
			double[,] A = Utilities.DistributeMatrixStriped(comm, TestProblem.MatrixA);
			double[] b = TestProblem.VectorAx;
			double[] expected = TestProblem.VectorX;
			double[] x = new double[TestProblem.Size];

			int numIterations = 100;
			double tolerance = 1E-7;
			IterativeMethodsMpi.SolvePCG(comm, A, b, x, numIterations, tolerance);

			Utilities.AssertEqual(expected, x, 5);
			Console.WriteLine($"Process {comm.Rank}: Test passed");
		}
    }
}
