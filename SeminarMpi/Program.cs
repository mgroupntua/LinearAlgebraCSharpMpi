using System.Text;
using System.Xml.Linq;
using MPI;
using MSolve.Edu.LinearAlgebra;
using SeminarMpi.LinearAlgebra;
using SeminarMpi.Tests;
using SeminarMpi.Utilities;

namespace SeminarMpi
{
    public  class Program
	{
		public static void Main(string[] args)
		{
			HelloWorld(args);

			//TransferTests.GatherVector(args);
			//TransferTests.GatherMatrix(args);
			//TransferTests.ScatterVector(args);
			//TransferTests.ScatterMatrix(args);

			//BlasTests.TestAxpby(args);
			//BlasTests.TestDotProduct(args);
			//BlasTests.TestInvertDiagonal(args);
			//BlasTests.TestMatrixVectorMult(args);

			//SolverTests.TestPcgSolverSerial(args);
			//SolverTests.TestPcgSolverMpi(args);

			//FemMpiTest.Run(args);
		}

		public static void HelloWorld(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				MpiUtilities.AssistDebuggerAttachment(comm);

				int rank = comm.Rank;
				string name = "User" + comm.Rank;
				comm.Barrier();
				Console.WriteLine($"MPI process {comm.Rank}: Hello user {name}");
				comm.Barrier();
			}
		}
	}
}