using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MPI;

namespace LinAlgMpi.LinearAlgebra
{
	public static class DistributedUtilities
	{
		public static void CopyDistributedVector(Intracommunicator comm, double[] source, double[] destination)
		{
			if (source.Length != destination.Length) throw new ArgumentException("The distributed vectors have different format");
			Array.Copy(source, destination, source.Length);
		}

		public static double[] CreateDistributedVectorWithSameFormat(Intracommunicator comm, double[] localVector)
		{
			return new double[localVector.Length];
		}

		public static void DistributedToMirrorVectorOLD(Intracommunicator comm, double[] globalVector, double[] localVector)
		{
			int globalSize = globalVector.Length;
			int numProcesses = comm.Size;
			int chunkSize = (globalSize - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

			// Define how many entries each local vector has
			int[] numLocalEntries = new int[numProcesses];
			Array.Fill(numLocalEntries, chunkSize);
			//numLocalEntries[numProcesses - 1] = globalSize - chunkSize * (numProcesses - 1); // Adjust the last process's contribution

			// Gather all local vector to the root process and then broadcast to the rest
			int root = 0;
			double[] buffer = new double[chunkSize * numProcesses]; // temporary buffer
			comm.GatherFlattened(localVector, numLocalEntries, root, ref buffer);
			comm.Broadcast(ref buffer, root);

			// Copy the active part of the buffer to the global vector
			Array.Copy(buffer, globalVector, globalSize);
		}

		public static void DistributedToMirrorVector(Intracommunicator comm, double[] globalVector, double[] localVector)
		{
			int globalSize = globalVector.Length;
			int numProcesses = comm.Size;
			int chunkSize = (globalSize - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

			// Define how many entries each local vector has
			int[] numLocalEntries = new int[numProcesses];
			Array.Fill(numLocalEntries, chunkSize);
			numLocalEntries[numProcesses - 1] = globalSize - chunkSize * (numProcesses - 1); // Adjust the last process's contribution

			// Gather all local vector to the root process and then broadcast to the rest
			int root = 0;
			comm.GatherFlattened(localVector, numLocalEntries, root, ref globalVector);
			comm.Broadcast(ref globalVector, root);
		}

		public static double[] MirroredToDistributedVector(Intracommunicator comm, double[] globalVector)
		{
			int globalLength = globalVector.Length;
			int numProcesses = comm.Size;
			int chunkSize = (globalLength - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

			int start = chunkSize * comm.Rank;
			int end = Math.Min(start + chunkSize, globalLength); // exclusive. Last process may get a smaller subvector

			// Copy the relevant entries of this process to a new array
			double[] localVector = new double[end - start];
			Array.Copy(globalVector, start, localVector, 0, end - start);

			return localVector;
		}

		public static double[,] MirroredToStripedMatrix(Intracommunicator comm, double[,] globalMatrix)
		{
			int numRows = globalMatrix.GetLength(0);
			int numColumns = globalMatrix.GetLength(1);
			int numProcesses = comm.Size;
			int chunkSize = (numRows - 1) / numProcesses + 1; // CEILING(numEntries / numThreads)

			// Copy the relevant entries of this process to a new array
			int startRow = chunkSize * comm.Rank;
			int endRow = Math.Min(startRow + chunkSize, numRows); // exclusive
			double[,] localMatrix = new double[endRow-startRow, numColumns];
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
