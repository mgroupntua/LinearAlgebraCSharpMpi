using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Xunit;

namespace LinAlgMPI
{
    public class Test
    {
        [Fact]
        public static void HelloWord()
        {
            string user = "User single";
            Debug.WriteLine($"Hello {user}");
        }
    }
}
