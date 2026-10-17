using System;
using LinAlgMpi.LinearAlgebra;
using LinAlgMpi.Tests;
using MPI;
using static System.Collections.Specialized.BitVector32;

namespace LinAlgMPI
{
	class Program
	{
		static void Main(string[] args)
		{
			//NormalTests.TestPcgNoMpi();

			//HelloWorld(args);
			RunMpiTests(debug: false, args, MpiTests.TestAxpbyMirror, MpiTests.TestAxpbyDistributed, MpiTests.TestDotProductMirror, MpiTests.TestDotProductDistributed, MpiTests.TestMultiplyMVMirrorStriped, MpiTests.TestSolveJacobi, MpiTests.TestSolvePCG);
		}

		public static void HelloWorld(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				Utilities.AssistDebuggerAttachment(comm);

				int rank = comm.Rank;
				string name = "User" + comm.Rank;
				comm.Barrier();
				Console.WriteLine($"MPI process {comm.Rank}: Hello user {name}");
				comm.Barrier();
			}
		}

		public static void RunMpiTests(bool debug, string[] args, params Action<Intracommunicator>[] tests)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				if (debug) Utilities.AssistDebuggerAttachment(comm);

				foreach (Action<Intracommunicator> test in tests)
				{
					if (comm.Rank == 0)
					{
						string className = test.Method.DeclaringType?.FullName;
						Console.WriteLine($"{className}.{test.Method.Name}:");
					}
					test(comm);
					comm.Barrier();
				}
			}
		}
	}
}
