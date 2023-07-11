using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MSolve.Edu.Analyzers;
using MSolve.Edu.FEM.Elements;
using MSolve.Edu.FEM.Entities;
using MSolve.Edu.FEM;
using MSolve.Edu.Solvers;
using MPI;
using SeminarMpi.LinearAlgebra;
using SeminarMpi.Utilities;

namespace SeminarMpi.Tests
{
	public class FemMpiTest
	{
		public static void Run(string[] args)
		{
			using (new MPI.Environment(ref args))
			{
				Intracommunicator comm = Communicator.world;
				MpiUtilities.AssistDebuggerAttachment(comm);

				// Create the FEM model and linear system, only in the master process
				Model model = null;
				int n = -1;
				double[] A = null; // Row major format
				double[] b = null;
				if (comm.Rank == Constants.masterProcess)
				{
					model = CreateModel();
					SkylineLinearSystem linearSystem = BuildLinearSystem(model);
					(n, A, b) = ConvertLinearSystem(linearSystem);
				}

				// Transfer to other processes the size of the linear system and the corresponding subvectors and submatrices
				comm.Broadcast(ref n, Constants.masterProcess);
				double[] subB = DataTransfers.ScatterVector(comm, n, b);
				(int ms, int ns, double[] subA) = DataTransfers.ScatterMatrix(comm, n, n, A);

				// Solve the linear system using MPI
				double[] subX = MpiBLAS.CreateZeroVector(comm, n);
				PcgSolver.SolveMpi(comm, n, subA, subB, subX, 1000, 1E-8);

				// Gather the solution vector to the master process from all others
				double[] x = DataTransfers.GatherVector(comm, n, subX);

				// Print results in master process only
				if (comm.Rank == Constants.masterProcess)
				{
					var msg = new StringBuilder();
					msg.AppendLine($"Process {comm.Rank}:");

					// Get the expected solution, by solving with MSolve.Edu tools
					double[] xExpected = BuildAndSolveLinearSystem(model);
					msg.AppendLine("expected: ");
					msg.AppendLine(MatrixOperations.VectorToString(xExpected));

					msg.AppendLine("computed: ");
					msg.AppendLine(MatrixOperations.VectorToString(x));
					Console.WriteLine(msg);
				}
			}
		}

		public static Model CreateModel()
		{
			Model model = new Model();

			// Nodes
			model.NodesDictionary[0] = new Node { ID = 0, X = 0.0, Y = 0.0 };
			model.NodesDictionary[1] = new Node { ID = 1, X = 0.0, Y = 0.5 };
			model.NodesDictionary[2] = new Node { ID = 2, X = 0.5, Y = 0.0 };
			model.NodesDictionary[3] = new Node { ID = 3, X = 0.5, Y = 0.5 };
			model.NodesDictionary[4] = new Node { ID = 4, X = 1.0, Y = 0.0 };
			model.NodesDictionary[5] = new Node { ID = 5, X = 1.0, Y = 0.5 };
			model.NodesDictionary[6] = new Node { ID = 6, X = 1.5, Y = 0.0 };
			model.NodesDictionary[7] = new Node { ID = 7, X = 1.5, Y = 0.5 };
			model.NodesDictionary[8] = new Node { ID = 8, X = 2.0, Y = 0.0 };
			model.NodesDictionary[9] = new Node { ID = 9, X = 2.0, Y = 0.5 };

			// Elements
			int numElements = 17;
			double E = 1E7;
			double A = 0.9;
			for (int e = 0; e < numElements; e++)
			{
				model.ElementsDictionary[e] = new Element() { ID = e };
				model.ElementsDictionary[e].ElementType = new Truss2D(E) { SectionArea = A };
			}

			// Element-node connectivity
			int[,] nodesOfElements = new int[,]
			{
				{ 0, 2 },
				{ 2, 4 },
				{ 4, 6 },
				{ 6, 8 },
				{ 1, 3 },
				{ 3, 5 },
				{ 5, 7 },
				{ 7, 9 },
				{ 0, 1 },
				{ 0, 3 },
				{ 2, 3 },
				{ 2, 5 },
				{ 4, 5 },
				{ 6, 5 },
				{ 6, 7 },
				{ 8, 7 },
				{ 8, 9 },
			};
			for (int e = 0; e < numElements; e++)
			{
				Node start = model.NodesDictionary[nodesOfElements[e, 0]];
				Node end = model.NodesDictionary[nodesOfElements[e, 1]];
				model.ElementsDictionary[e].AddNode(start);
				model.ElementsDictionary[e].AddNode(end);
			}

			// Constrain dofs at support locations 
			model.NodesDictionary[0].Constraints.AddRange(new[] { DOFType.X, DOFType.Y });
			model.NodesDictionary[8].Constraints.AddRange(new[] { DOFType.X, DOFType.Y });

			// Apply nodal loads
			model.Loads.Add(new Load() { Amount = -1000, Node = model.NodesDictionary[5], DOF = DOFType.Y });

			// Finalize
			model.ConnectDataStructures();
			return model;
		}

		public static (int n, double[] A, double[] b) ConvertLinearSystem(SkylineLinearSystem linearSystem)
		{
			double[] b = linearSystem.RHS.Data;
			int n = b.Length;
			double[] A = MatrixOperations.ConvertToRowMajor(linearSystem.Matrix);
			return (n, A, b);
		}

		public static SkylineLinearSystem BuildLinearSystem(Model model)
		{
			// Default solver in MSolve.Edu is LDL factorization in Skyline format
			var linearSystem = new SkylineLinearSystem(model.Forces);
			var solver = new SolverSkyline(linearSystem);

			// Linear static analysis for structural problem (other problems could be thermal, thermomechanical, etc)
			var provider = new ProblemStructural(model);
			var childAnalyzer = new LinearAnalyzer(solver);
			var parentAnalyzer = new StaticAnalyzer(provider, childAnalyzer, linearSystem);

			parentAnalyzer.BuildMatrices();
			return linearSystem;
		}

		public static double[] BuildAndSolveLinearSystem(Model model)
		{
			// Default solver in MSolve.Edu is LDL factorization in Skyline format
			var linearSystem = new SkylineLinearSystem(model.Forces);
			var solver = new SolverSkyline(linearSystem);

			// Linear static analysis for structural problem (other problems could be thermal, thermomechanical, etc)
			var provider = new ProblemStructural(model);
			var childAnalyzer = new LinearAnalyzer(solver);
			var parentAnalyzer = new StaticAnalyzer(provider, childAnalyzer, linearSystem);

			parentAnalyzer.BuildMatrices();
			parentAnalyzer.Initialize();
			parentAnalyzer.Solve();

			return linearSystem.Solution.Data;
		}
	}
}
