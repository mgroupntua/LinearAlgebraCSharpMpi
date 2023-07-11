using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MSolve.Edu.LinearAlgebra;

namespace SeminarMpi.LinearAlgebra
{
    public class MatrixOperations
    {
        public static double[,] ConvertToMatrix2D(int m, int n, double[] matrix)
        {
            double[,] result = new double[m, n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    result[i, j] = matrix[i * n + j];
                }
            }
            return result;
        }

        public static double[] ConvertToRowMajor(double[,] matrix)
        {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
            double[] result = new double[m * n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    result[i * n + j] = matrix[i, j];
                }
            }
            return result;
        }

        public static double[] ConvertToRowMajor(SkylineMatrix2D matrix)
        {
            int m = matrix.Rows;
            int n = matrix.Columns;
            double[] result = new double[m * n];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    result[i * n + j] = matrix[i, j];
                }
            }
            return result;
        }

        public static string VectorToString(double[] vector)
        {
            var msg = new StringBuilder();
            for (int i = 0; i < vector.Length; i++)
            {
                msg.Append(" ");
                msg.Append(vector[i]);
            }
            return msg.ToString();
        }

		public static string MatrixToString(double[,] matrix)
		{
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
			var msg = new StringBuilder();
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < n; j++)
				{
					msg.Append(" ");
					msg.Append(matrix[i, j]);
				}
				msg.AppendLine();
			}
			return msg.ToString();
		}

		public static string MatrixToString(int m, int n, double[] matrixRowMajor)
        {
            var msg = new StringBuilder();
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    msg.Append(" ");
                    msg.Append(matrixRowMajor[i * n + j]);
                }
				msg.AppendLine();
			}
			return msg.ToString();
        }
    }
}
