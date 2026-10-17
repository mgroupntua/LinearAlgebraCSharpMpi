using System;
using System.Collections.Generic;
using System.Text;

namespace LinAlgMpi.Tests
{
	public static class TestProblem
	{
		public static double[,] MatrixA => new double[,]
		{
			{ 21.0,  1.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },
			{  1.0, 22.0,  2.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0 },
			{  0.0,  2.0, 23.0,  1.0,  3.0,  1.0,  0.0,  1.0,  0.0,  0.0 },
			{  4.0,  0.0,  1.0, 24.0,  2.0,  4.0,  0.0,  0.0,  0.0,  0.0 },
			{  0.0,  0.0,  3.0,  2.0, 25.0,  5.0,  2.0,  0.0,  0.0,  1.0 },
			{  0.0,  0.0,  1.0,  4.0,  5.0, 26.0,  0.0,  0.0,  2.0,  3.0 },
			{  0.0,  1.0,  0.0,  0.0,  2.0,  0.0, 27.0,  3.0,  0.0,  0.0 },
			{  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  3.0, 28.0,  4.0,  2.0 },
			{  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  4.0, 29.0,  0.0 },
			{  0.0,  0.0,  0.0,  0.0,  1.0,  3.0,  0.0,  2.0,  0.0, 30.0 }
		};

		public static double[] VectorAx => new double[] { 39, 58, 106, 137, 196, 248, 225, 304, 305, 339 };

		public static double[] VectorX => new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

		public static double[] VectorY => new double[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

		public static double[] Vector2X3Y => new double[] { 32, 64, 96, 128, 160, 192, 224, 256, 288, 320 };

		public const double dotXY = 3850;

		public const int Size = 10;
	}
}
