using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace MLConsoleAP
{
    public static class ShuffleRows
    {
        public static void LottoDemo()
        {
            var mlContext = new MLContext();
            var listData = GetLottoData();
            var data = mlContext.Data.LoadFromEnumerable(listData);
            var shuffledData = mlContext.Data.ShuffleRows(data, seed: 123);
            var enumerable = mlContext.Data
                .CreateEnumerable<LottoData>(shuffledData,
                reuseRowObject: true);
            foreach (var row in enumerable) 
            {
                Console.WriteLine($"date: {row.Date.ToString()}, number: {row.ToString()}");
            }

        }

        public static int[] GetRandomNumbers()
        {
            List<int> numbers = new List<int>();
            var random = new Random();
            numbers.Add(random.Next(1, 50));
            numbers.Add(random.Next(1, 50));
            numbers.Add(random.Next(1, 50));
            numbers.Add(random.Next(1, 50));
            numbers.Add(random.Next(1, 50));
            numbers.Add(random.Next(1, 50));

            return numbers.ToArray();
        }

        public static int[] GetRandomNumbers(int length,int start,int end)
        {
            List<int> numbers = new List<int>();
            var random = new Random();

            for (int i = 0; i < length; i++)
            {
                int number = random.Next(start, end);
                while(numbers.Contains(number))
                {
                    number = random.Next(start, end);
                }
                numbers.Add(number);
            }

            return numbers.ToArray();
        }

        public static void PrintRandomNumbers()
        {
            var numbers = GetRandomNumbers(6, 1, 50);
            string str = string.Empty;
            foreach (var num in numbers)
            {
                str += num.ToString() + " ";
            }
            Console.WriteLine(str);
        }

        public static void PrintRandomNumbers2()
        {
            var numbers = GetRandomNumbers(12, 1, 25);
            string str = string.Empty;
            foreach (var num in numbers)
            {
                str += num.ToString() + " ";
            }
            Console.WriteLine(str);
        }

        // Sample class showing how to shuffle rows in 
        // IDataView.
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            var enumerableOfData = GetSampleTemperatureData(5);
            var data = mlContext.Data.LoadFromEnumerable(enumerableOfData);

            // Before we apply a filter, examine all the records in the dataset.
            Console.WriteLine($"Date\tTemperature");
            foreach (var row in enumerableOfData)
            {
                Console.WriteLine($"{row.Date.ToString("d")}" +
                    $"\t{row.Temperature}");
            }
            Console.WriteLine();
            // Expected output:
            //  Date    Temperature
            //  1/2/2012        36
            //  1/3/2012        36
            //  1/4/2012        34
            //  1/5/2012        35
            //  1/6/2012        35

            // Shuffle the dataset.
            var shuffledData = mlContext.Data.ShuffleRows(data, seed: 123);

            // Look at the shuffled data and observe that the rows are in a
            // randomized order.
            var enumerable = mlContext.Data
                .CreateEnumerable<SampleTemperatureData>(shuffledData,
                reuseRowObject: true);

            Console.WriteLine($"Date\tTemperature");
            foreach (var row in enumerable)
            {
                Console.WriteLine($"{row.Date.ToString("d")}" +
                $"\t{row.Temperature}");
            }
            // Expected output:
            //  Date    Temperature
            //  1/4/2012        34
            //  1/2/2012        36
            //  1/5/2012        35
            //  1/3/2012        36
            //  1/6/2012        35
        }

        private class SampleTemperatureData
        {
            public DateOnly Date { get; set; }
            public float Temperature { get; set; }
        }

        public class LottoData
        {
            public DateTime Date { get; set; } = DateTime.Now;
            public int[]? Numbers { get; set; } = null;

            public override string ToString()
            {
                string str = string.Empty;
                foreach (var num in Numbers)
                {
                    str += num.ToString() + " ";
                }
                return str;
            }
        }

        /// <summary>
        /// Get a fake temperature dataset.
        /// </summary>
        /// <param name="exampleCount">The number of examples to return.</param>
        /// <returns>An enumerable of <see cref="SampleTemperatureData"/>.</returns>
        private static IEnumerable<SampleTemperatureData> GetSampleTemperatureData(
            int exampleCount)

        {
            var rng = new Random(1234321);
            var date = new DateOnly(2012, 1, 1);
            float temperature = 39.0f;

            for (int i = 0; i < exampleCount; i++)
            {
                date = date.AddDays(1);
                temperature += rng.Next(-5, 5);
                yield return new SampleTemperatureData
                {
                    Date = date,
                    Temperature =
                    temperature
                };

            }
        }

        public static List<LottoData> GetLottoData()
        {
            List<LottoData> list = new List<LottoData>();
            list.Add(new LottoData
            {
                Date = new DateTime(2023, 3, 7),
                Numbers = new int[] { 13, 17, 23, 25, 32, 41 },
            });
            list.Add(new LottoData
            {
                Date = new DateTime(2023, 3, 3),
                Numbers = new int[] { 13, 16, 18, 22, 34, 42 },
            });
            list.Add(new LottoData
            {
                Date = new DateTime(2023, 2, 28),
                Numbers = new int[] { 02, 11, 25, 26, 27, 35 },
            });
            list.Add(new LottoData
            {
                Date = new DateTime(2023, 2, 24),
                Numbers = new int[] { 15, 38, 39, 40, 45, 48 },
            });
            list.Add(new LottoData
            {
                Date = new DateTime(2023, 2, 21),
                Numbers = new int[] { 01, 05, 06, 11, 26, 46 },
            });
            list.Add(new LottoData
            {
                Date = new DateTime(2023, 2, 17),
                Numbers = new int[] { 07, 09, 14, 16, 26, 33 },
            });

            return list;
        }
    }
}
