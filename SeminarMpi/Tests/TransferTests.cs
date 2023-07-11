using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using MPI;
using SeminarMpi.LinearAlgebra;
using SeminarMpi.Utilities;

namespace SeminarMpi.Tests
{
	public class TransferTests
	{
		public static void GatherVector(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				int n = TestData.n;
				double[] subX = TestData.GetSubX(comm.Rank);

				double[] x = DataTransfers.GatherVector(comm, n, subX);

				if (comm.Rank == Constants.masterProcess)
				{
					var msg = new StringBuilder();
					msg.AppendLine($"Process {comm.Rank}:");
					msg.AppendLine("expected: ");
					msg.AppendLine(MatrixOperations.VectorToString(TestData.x));
					msg.AppendLine("computed: ");
					msg.AppendLine(MatrixOperations.VectorToString(x));
					Console.WriteLine(msg);
				}
			}
		}

		public static void GatherMatrix(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				int m = TestData.n;
				int n = TestData.n;
				double[,] submatrix = TestData.GetSubA(comm.Rank);
				double[] subA = MatrixOperations.ConvertToRowMajor(submatrix);

				double[] A = DataTransfers.GatherMatrix(comm, m, n, subA);

				if (comm.Rank == Constants.masterProcess)
				{
					var msg = new StringBuilder();
					msg.AppendLine($"Process {comm.Rank}:");
					msg.AppendLine("expected: ");
					msg.AppendLine(MatrixOperations.MatrixToString(TestData.A));
					msg.AppendLine("computed: ");
					msg.AppendLine(MatrixOperations.MatrixToString(m, n, A));
					Console.WriteLine(msg);
				}
			}
		}

		public static void ScatterVector(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				int n = TestData.n;
				double[] x = null;
				if (comm.Rank == Constants.masterProcess)
				{
					x = TestData.x;
				}

				double[] subX = DataTransfers.ScatterVector(comm, n, x);

				var msg = new StringBuilder();
				msg.AppendLine($"Process {comm.Rank}:");
				msg.AppendLine("expected: ");
				msg.AppendLine(MatrixOperations.VectorToString(TestData.GetSubX(comm.Rank)));
				msg.AppendLine("computed: ");
				msg.AppendLine(MatrixOperations.VectorToString(subX));
				Console.WriteLine(msg);
			}
		}

		public static void ScatterMatrix(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				int m = TestData.n;
				int n = TestData.n;
				double[] A = null;
				if (comm.Rank == Constants.masterProcess)
				{
					double[,] matrix = TestData.A;
					A = MatrixOperations.ConvertToRowMajor(matrix);
				}

				(int ms, int ns, double[] subA) = DataTransfers.ScatterMatrix(comm, m, n, A);

				var msg = new StringBuilder();
				msg.AppendLine($"Process {comm.Rank}:");
				msg.AppendLine("expected: ");
				msg.AppendLine(MatrixOperations.MatrixToString(TestData.GetSubA(comm.Rank)));
				msg.AppendLine("computed: ");
				msg.AppendLine(MatrixOperations.MatrixToString(ms, ns, subA));
				Console.WriteLine(msg);
				//MpiUtilities.DoSerially(comm, () => Console.WriteLine(msg));
			}
		}
	}
}
