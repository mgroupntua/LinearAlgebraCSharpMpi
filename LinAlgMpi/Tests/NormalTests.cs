using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using LinAlgMpi.LinearAlgebra;
using Xunit;

namespace LinAlgMpi.Tests
{
    public class NormalTests
    {
		public static void TestPcgSerial()
        {
			double[,] A = TestProblem.MatrixA;
			double[] b = TestProblem.VectorAx;
			double[] expected = TestProblem.VectorX;
			double[] x = new double[TestProblem.Size];

			int numIterations = 100;
			double tolerance = 1E-7;
			IterativeMethodsSerial.SolvePCG(A, b, x, numIterations, tolerance);

			Utilities.AssertEqual(expected, x, 5);
			Console.WriteLine("Test passed");
		}

		public static void TestPcgShared()
		{
			double[,] A = TestProblem.MatrixA;
			double[] b = TestProblem.VectorAx;
			double[] expected = TestProblem.VectorX;
			double[] x = new double[TestProblem.Size];

			int numIterations = 100;
			double tolerance = 1E-7;
			IterativeMethodsShared.SolvePCG(A, b, x, numIterations, tolerance);

			Utilities.AssertEqual(expected, x, 5);
			Console.WriteLine("Test passed");
		}
	}
}
