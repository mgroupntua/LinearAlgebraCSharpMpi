using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MPI;
using LinAlgMPI.LinearAlgebra;
using Xunit;

namespace LinAlgMPI.Tests
{
    public class MpiTests
    {
        public static double[,] MatrixA => new double[,] 
        { 
            { 21.0,  1.0,  0.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0 },
            {  1.0, 22.0,  2.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0 },
            {  0.0,  2.0, 23.0,  1.0,  3.0,  1.0,  0.0,  1.0,  0.0,  0.0 },
            {  4.0,  0.0,  1.0, 24.0,  2.0,  4.0,  0.0,  0.0,  0.0,  0.0 },
            {  0.0,  0.0,  3.0,  2.0, 25.0,  5.0,  2.0,  0.0,  0.0,  1.0 },
            {  0.0,  0.0,  1.0,  4.0,  5.0, 26.0,  0.0,  0.0,  2.0,  3.0 },
            {  0.0,  1.0,  0.0,  0.0,  2.0,  0.0, 27.0,  3.0,  0.0,  0.0 },
            {  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  3.0, 28.0,  4.0,  2.0 },
            {  0.0,  0.0,  0.0,  0.0,  0.0,  2.0,  0.0,  4.0, 29.0,  0.0 },
            {  0.0,  0.0,  0.0,  0.0,  1.0,  3.0,  0.0,  2.0,  0.0, 30.0 } 
        };

        public static double[] VectorAx => new double[] { 39, 58, 106, 137, 196, 248, 225, 304, 305, 339 };

        public static double[] VectorX => new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        public static double[] VectorY => new double[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

        public static double[] Vector2X3Y => new double[] { 32, 64, 96, 128, 160, 192, 224, 256, 288, 320 };

        public const double dotXY = 3850;

        public const int Size = 10;

        //public static void TestAxpbyMirror(string[] args)
        //{
        //    using (new MPI.Environment(ref args))
        //    {
        //        Intracommunicator comm = Communicator.world;
        //        MPIUtilities.AssistDebuggerAttachment(comm);

        //        double[] x = VectorX;
        //        double[] y = VectorY;
        //        double[] result = new double[Size];
        //        MpiBLAS.AxpbyMirror(comm, Size, 2, x, 3, y, result);
        //        double[] expected = Vector2X3Y;

        //        AssertEqual(expected, result, 5);
        //        Console.WriteLine($"Process {comm.Rank}: Test passed");
        //    }
        //}

        public static void TestAxpbyDistributed(string[] args)
        {
            using (new MPI.Environment(ref args))
            {
                Intracommunicator comm = Communicator.world;
                MPIUtilities.AssistDebuggerAttachment(comm);

                double[] x = MPIUtilities.DistributeVector(comm, VectorX);
                double[] y = MPIUtilities.DistributeVector(comm, VectorY);
                double[] result = MPIUtilities.DistributeVector(comm, new double[Size]);

                MpiBLAS.AxpbyDistributed(comm, Size, 2, x, 3, y, result);
                double[] expected = MPIUtilities.DistributeVector(comm, Vector2X3Y);

                AssertEqual(expected, result, 5);
                Console.WriteLine($"Process {comm.Rank}: Test passed");
            }
        }

        //public static void TestDotProductMirror(string[] args)
        //{
        //    // This code will be run on every process.
        //    using (new MPI.Environment(ref args))
        //    {
        //        Intracommunicator comm = Communicator.world;
        //        MPIUtilities.AssistDebuggerAttachment(comm);

        //        // Each process will own the whole vector, but only operate on a subvector.
        //        double[] x = VectorX;
        //        double[] y = VectorY;

        //        double result = MpiBLAS.DotProductMirror(comm, Size, x, y);
        //        double expected = dotXY;

        //        Assert.Equal(expected, result, 5);
        //        Console.WriteLine($"Process {comm.Rank}: Test passed");
        //    }
        //}

        //public static void TestDotProductDistributed(string[] args)
        //{
        //    // This code will be run on every process.
        //    using (new MPI.Environment(ref args))
        //    {
        //        Intracommunicator comm = Communicator.world;
        //        MPIUtilities.AssistDebuggerAttachment(comm);

        //        // Each process will own only a part of the whole vector.
        //        double[] x = MPIUtilities.DistributeVector(comm, VectorX);
        //        double[] y = MPIUtilities.DistributeVector(comm, VectorY);

        //        double result = MpiBLAS.DotProductDistributed(comm, Size, x, y);
        //        double expected = dotXY;

        //        Assert.Equal(expected, result, 5);
        //        Console.WriteLine($"Process {comm.Rank}: Test passed");
        //    }
        //}

        public static void TestMultiplyMVMirrorStriped(string[] args)
        {
            // This code will be run on every process.
            using (new MPI.Environment(ref args))
            {
                Intracommunicator comm = Communicator.world;
                MPIUtilities.AssistDebuggerAttachment(comm);

                // Each process will own only a part of the whole vector.
                double[,] A = MPIUtilities.DistributeMatrixStriped(comm, MatrixA);
                double[] x = VectorX;

                double[] result = new double[Size];
                MpiBLAS.MultiplyMatrixVectorMirrorStriped(comm, Size, Size, A, x, result);
                double[] expected = VectorAx;

                AssertEqual(expected, result, 5);
                Console.WriteLine($"Process {comm.Rank}: Test passed");
            }
        }

        //public static void TestSolveJacobi(string[] args)
        //{
        //    // This code will be run on every process.
        //    using (new MPI.Environment(ref args))
        //    {
        //        Intracommunicator comm = Communicator.world;
        //        MPIUtilities.AssistDebuggerAttachment(comm);

        //        double[,] A = MPIUtilities.DistributeMatrixStriped(comm, MatrixA);
        //        double[] b = VectorAx;
        //        double[] expected = VectorX;
        //        double[] x = new double[Size];

        //        int numIterations = 100;
        //        double tolerance = 1E-7;
        //        IterativeMethodsMpi.SolveJacobi(comm, A, b, x, numIterations, tolerance);

        //        AssertEqual(expected, x, 3);
        //        Console.WriteLine($"Process {comm.Rank}: Test passed");
        //    }
        //}

        //public static void TestSolvePCG(string[] args)
        //{
        //    // This code will be run on every process.
        //    using (new MPI.Environment(ref args))
        //    {
        //        Intracommunicator comm = Communicator.world;
        //        MPIUtilities.AssistDebuggerAttachment(comm);

        //        double[,] A = MPIUtilities.DistributeMatrixStriped(comm, MatrixA);
        //        double[] b = VectorAx;
        //        double[] expected = VectorX;
        //        double[] x = new double[Size];

        //        int numIterations = 100;
        //        double tolerance = 1E-7;
        //        IterativeMethodsMpi.SolvePCG(comm, A, b, x, numIterations, tolerance);

        //        AssertEqual(expected, x, 5);
        //        Console.WriteLine($"Process {comm.Rank}: Test passed");
        //    }
        //}

        private static void AssertEqual(double[] expected, double[] actual, int precision)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i], precision);
            }
        }
    }
}
