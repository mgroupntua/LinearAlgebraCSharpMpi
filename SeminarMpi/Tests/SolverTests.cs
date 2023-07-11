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
	public class SolverTests
	{
		public static void TestPcgSolverSerial(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				//MpiUtilities.AssistDebuggerAttachment(comm);

				if (comm.Rank == Constants.masterProcess)
				{
					int n = TestData.n;
					double[] A = MatrixOperations.ConvertToRowMajor(TestData.A);
					double[] y = TestData.y;

					double[] x = new double[n];
					PcgSolver.SolveSerial(n, A, y, x, 1000, 1E-8);

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

		public static void TestPcgSolverMpi(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				MpiUtilities.AssistDebuggerAttachment(comm);

				int n = TestData.n;
				double[] A = MatrixOperations.ConvertToRowMajor(TestData.GetSubA(comm.Rank));
				double[] y = TestData.GetSubY(comm.Rank);

				double[] x = MpiBLAS.CreateZeroVector(comm, n);
				PcgSolver.SolveMpi(comm, n, A, y, x, 1000, 1E-8);

				var msg = new StringBuilder();
				msg.AppendLine($"Process {comm.Rank}:");
				msg.AppendLine("expected: ");
				msg.AppendLine(MatrixOperations.VectorToString(TestData.GetSubX(comm.Rank)));
				msg.AppendLine("computed: ");
				msg.AppendLine(MatrixOperations.VectorToString(x));
				Console.WriteLine(msg);
			}
		}
	}
}
