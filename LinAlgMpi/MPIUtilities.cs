﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MPI;

namespace LinAlgMPI
{
	public static class MPIUtilities
	{
		public static void AssistDebuggerAttachment(Intracommunicator comm)
		{
			DoSerially(comm, () => Console.WriteLine($"MPI process {comm.Rank}: PID = {Process.GetCurrentProcess().Id}"));
			if (comm.Rank == 0)
			{
				Console.Write("All processes of the application have paused.");
				Console.Write(" While in this state you can optionally attach a debugger.");
				Console.WriteLine(" After you have finished, type anything to continue.");
				Console.ReadLine();
			}
			comm.Barrier();
		}

		public static void DoSerially(Intracommunicator comm, Action action)
		{
			comm.Barrier();
			int token = 0;
			if (comm.Rank == 0)
			{
				// Perform action in current process only
				action();

				// Send token to our right neighbor
				comm.Send(token, (comm.Rank + 1) % comm.Size, 0);

				// Receive token from our left neighbor
				comm.Receive((comm.Rank + comm.Size - 1) % comm.Size, 0, out token);
			}
			else
			{
				// Receive token from our left neighbor
				comm.Receive((comm.Rank + comm.Size - 1) % comm.Size, 0, out token);

				// Perform action in current process only
				action();

				// Pass on the token to our right neighbor
				comm.Send(token, (comm.Rank + 1) % comm.Size, 0);
			}
			comm.Barrier();
		}

		public static double[] DistributeVector(Intracommunicator comm, double[] globalVector)
		{
			int globalLength = globalVector.Length;
			int numProcesses = comm.Size;
			int chunkSize = (globalLength - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

			// Copy the relevant entries of this process to a new array
			double[] localVector = new double[chunkSize];
			int start = chunkSize * comm.Rank;
			int end = Math.Min(start + chunkSize, globalLength); // exclusive
			Array.Copy(globalVector, start, localVector, 0, end - start);

			return localVector;
		}

		public static double[,] DistributeMatrixStriped(Intracommunicator comm, double[,] globalMatrix)
		{
			int numRows = globalMatrix.GetLength(0);
			int numColumns = globalMatrix.GetLength(1);
			int numProcesses = comm.Size;
			int chunkSize = (numRows - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

			// Copy the relevant entries of this process to a new array
			double[,] localMatrix = new double[chunkSize, numColumns];
			int startRow = chunkSize * comm.Rank;
			int endRow = Math.Min(startRow + chunkSize, numRows); // exclusive
			for (int i = 0; i < endRow - startRow; i++)
			{
				for (int j = 0; j < numColumns; ++j)
				{
					localMatrix[i, j] = globalMatrix[startRow + i, j];
				}
			}

			return localMatrix;
		}
	}
}
