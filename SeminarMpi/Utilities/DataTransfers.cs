using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MPI;
using MSolve.Edu.LinearAlgebra;

namespace SeminarMpi.Utilities
{
    public class DataTransfers
    {
        public static int[] FindChunkSizes(int numProcesses, int numEntries)
        {
            int defaultSize = (numEntries - 1) / numProcesses + 1; // CEILING(numEntries / numProcesses)
            int[] chunkSizes = new int[numProcesses];
            for (int p = 0; p < numProcesses; p++)
            {
                if (p < numProcesses - 1)
                {
                    chunkSizes[p] = defaultSize;
                }
                else
                {
                    int start = p * defaultSize;
                    int end = numEntries - 1;
                    chunkSizes[p] = end - start + 1;
                }
            }
            return chunkSizes;
        }

        public static double[] GatherMatrix(Intracommunicator comm, int m, int n, double[] submatrix)
        {
            int numProcesses = comm.Size;
            int ns = n;
            int[] chunkSizes = FindChunkSizes(numProcesses, m);
            for (int p = 0; p < numProcesses; p++)
            {
                chunkSizes[p] *= ns;
            }
            double[] globalMatrix = comm.GatherFlattened(submatrix, chunkSizes, Constants.masterProcess);
            return globalMatrix;
        }

        public static double[] GatherVector(Intracommunicator comm, int n, double[] subvector)
        {
            int numProcesses = comm.Size;
            int[] chunkSizes = FindChunkSizes(numProcesses, n);
            double[] globalVector = comm.GatherFlattened(subvector, chunkSizes, Constants.masterProcess);
            return globalVector;
        }

        public static (int ms, int ns, double[] submatrix) ScatterMatrix(Intracommunicator comm, int m, int n, double[] globalMatrix)
        {
            int numProcesses = comm.Size;

            int[] chunkSizes = FindChunkSizes(numProcesses, m);
            int ms = chunkSizes[comm.Rank];
            int ns = n;
            for (int p = 0; p < numProcesses; p++)
            {
                chunkSizes[p] *= ns;
            }

            double[] submatrix = comm.ScatterFromFlattened(globalMatrix, chunkSizes, Constants.masterProcess);
            return (ms, ns, submatrix);
        }


        public static double[] ScatterVector(Intracommunicator comm, int n, double[] globalVector)
        {
            int numProcesses = comm.Size;
            int[] chunkSizes = FindChunkSizes(numProcesses, n);
            double[] subvector = comm.ScatterFromFlattened(globalVector, chunkSizes, Constants.masterProcess);
            return subvector;
        }
    }
}
