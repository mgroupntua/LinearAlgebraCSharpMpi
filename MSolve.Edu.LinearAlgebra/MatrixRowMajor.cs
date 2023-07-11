//using System;
//using System.Collections.Generic;
//using System.Text;

//namespace MSolve.Edu.LinearAlgebra
//{
//	public class MatrixRowMajor
//	{
//		public int NumRows { get; }

//		public int NumCols { get; }

//		public double[] Data { get; }

//		public double this[int i, int j] => Data[i * NumCols + j];

//		public MatrixRowMajor(int numRows, int numCols)
//		{
//			NumRows = numRows;
//			NumCols = numCols;
//			Data = new double[numRows * numCols];
//		}

//		public MatrixRowMajor(int numRows, int numCols, double[] data)
//		{
//			NumRows = numRows;
//			NumCols = numCols;
//			Data = data;
//		}

//		public MatrixRowMajor(double[,] matrix)
//		{
//			NumRows = matrix.GetLength(0);
//			NumCols = matrix.GetLength(1);
//			Data = new double[NumRows * NumCols];
//			for (int i = 0; i < NumRows; i++)
//			{
//				for (int j = 0; j < NumCols; j++)
//				{
//					Data[i * NumCols + j] = matrix[i, j];
//				}
//			}
//		}

//		public MatrixRowMajor(SkylineMatrix2D matrix)
//		{
//			NumRows = matrix.Rows;
//			NumCols = matrix.Columns;
//			Data = new double[NumRows * NumCols];
//			for (int i = 0; i < NumRows; i++)
//			{
//				for (int j = 0; j < NumCols; j++)
//				{
//					Data[i * NumCols + j] = matrix[i, j];
//				}
//			}
//		}

//		public double[] MultiplyVector(double[] vector)
//		{
//			double[] result = new double[NumRows];
//			for (int i = 0; i < NumRows; i++)
//			{
//				for (int j = 0; j < NumCols; j++)
//				{
//					result[i] += Data[i * NumCols + j] * vector[j];
//				}
//			}
//			return result;
//		}
//	}
//}
