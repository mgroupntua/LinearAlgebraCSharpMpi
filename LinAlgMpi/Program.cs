using System;
using LinAlgMPI.Tests;
using MPI;

namespace LinAlgMPI
{
	class Program
	{
		static void Main(string[] args)
		{
			//HelloWorld(args);
			//MpiTests.TestAxpbyDistributed(args);
			MpiTests.TestMultiplyMVMirrorStriped(args);
		}

		public static void HelloWorld(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				MPIUtilities.AssistDebuggerAttachment(comm);

				int rank = comm.Rank;
				string name = "User" + comm.Rank;
				comm.Barrier();
				Console.WriteLine($"MPI process {comm.Rank}: Hello user {name}");
				comm.Barrier();
			}
		}
	}
}
