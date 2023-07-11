using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MPI;
using SeminarMpi.LinearAlgebra;
using SeminarMpi.Utilities;

namespace SeminarMpi.Tests
{
	public class BlasTests
	{
		public static void TestAxpby(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				int n = TestData.n;
				double[] x = TestData.GetSubX(comm.Rank);
				double[] y = TestData.GetSubY(comm.Rank);

				double[] z = MpiBLAS.CreateZeroVector(comm, n);
				MpiBLAS.Axpby(comm, n, 2.0, x, 3.0, y, z);

				var msg = new StringBuilder();
				msg.AppendLine($"Process {comm.Rank}:");
				msg.AppendLine("expected: ");
				msg.AppendLine(MatrixOperations.VectorToString(TestData.GetSubZ(comm.Rank)));
				msg.AppendLine("computed: ");
				msg.AppendLine(MatrixOperations.VectorToString(z));
				Console.WriteLine(msg);
			}
		}

		public static void TestDotProduct(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				int n = TestData.n;
				double[] x = TestData.GetSubX(comm.Rank);
				double[] y = TestData.GetSubY(comm.Rank);

				double dot = MpiBLAS.DotProduct(comm, n, x, y);

				var msg = new StringBuilder();
				msg.AppendLine($"Process {comm.Rank}:");
				msg.AppendLine($"expected: {TestData.xDotY}");
				msg.AppendLine($"computed: {dot}");
				Console.WriteLine(msg);
			}
		}

		public static void TestInvertDiagonal(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				MpiUtilities.AssistDebuggerAttachment(comm);

				int n = TestData.n;
				double[] A = MatrixOperations.ConvertToRowMajor(TestData.GetSubA(comm.Rank));

				double[] invD = MpiBLAS.InvertDiagonal(comm, n, A);

				var msg = new StringBuilder();
				msg.AppendLine($"Process {comm.Rank}:");
				msg.AppendLine("expected: ");
				msg.AppendLine(MatrixOperations.VectorToString(TestData.GetSubInvD(comm.Rank)));
				msg.AppendLine("computed: ");
				msg.AppendLine(MatrixOperations.VectorToString(invD));
				Console.WriteLine(msg);
			}
		}

		public static void TestMatrixVectorMult(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				int n = TestData.n;
				double[] A = MatrixOperations.ConvertToRowMajor(TestData.GetSubA(comm.Rank));
				double[] x = TestData.GetSubX(comm.Rank);

				double[] y = MpiBLAS.CreateZeroVector(comm, n);
				MpiBLAS.MultiplyMatrixVector(comm, n, n, A, x, y);

				var msg = new StringBuilder();
				msg.AppendLine($"Process {comm.Rank}:");
				msg.AppendLine("expected: ");
				msg.AppendLine(MatrixOperations.VectorToString(TestData.GetSubY(comm.Rank)));
				msg.AppendLine("computed: ");
				msg.AppendLine(MatrixOperations.VectorToString(y));
				Console.WriteLine(msg);
			}
		}
	}
}
