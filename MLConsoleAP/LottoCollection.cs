using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLConsoleAP
{
    public class LotteryData
    {
        [LoadColumn(0)]
        public float Number1 { get; set; }

        [LoadColumn(1)]
        public float Number2 { get; set; }

        [LoadColumn(2)]
        public float Number3 { get; set; }

        [LoadColumn(3)]
        public float Number4 { get; set; }

        [LoadColumn(4)]
        public float Number5 { get; set; }

        [LoadColumn(5)]
        public float Number6 { get; set; }

        [LoadColumn(6)]
        [ColumnName("Label")]
        public float SpecialNumber { get; set; } = -1.5f;

        public override string ToString()
        {
            string result = $"Lotto Result: {Number1}, {Number2}, {Number3}, {Number4}, {Number5}, {Number6}, {SpecialNumber}";
            return result;
        }
    }

    public class LottoTool
    {
        public void TrainData()
        {
            var mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<LotteryData>("lotto.txt", separatorChar: ',', hasHeader: true);
            List<int[]> numbers = new List<int[]>();

            foreach (var row in mlContext.Data.CreateEnumerable<LotteryData>(dataView, reuseRowObject: true))
            {
                var numberArray = new int[] { (int)row.Number1, (int)row.Number2, (int)row.Number3, (int)row.Number4, (int)row.Number5, (int)row.Number6 };
                numbers.Add(numberArray);
            }

            //var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = mlContext.Transforms.Conversion
                                    .MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label")
                                    .Append(mlContext.Transforms.Concatenate("Features", nameof(LotteryData.Number1), nameof(LotteryData.Number2), nameof(LotteryData.Number3), nameof(LotteryData.Number4), nameof(LotteryData.Number5), nameof(LotteryData.Number6)))
                                    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                                    .Append(mlContext.Regression.Trainers.Sdca());

            //var model = pipeline.Fit(trainTestData.TrainSet);
            var model = pipeline.Fit(dataView);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<LotteryData, LotteryData>(model);

            var lastNumbers = numbers.Last();
            LotteryData lastData = new LotteryData
            {
                Number1 = lastNumbers[0],
                Number2 = lastNumbers[1],
                Number3 = lastNumbers[2],
                Number4 = lastNumbers[3],
                Number5 = lastNumbers[4],
                Number6 = lastNumbers[5],
                SpecialNumber = 0,
            };
            var predictedSpecialNumber = predictionEngine.Predict(lastData);
            Console.WriteLine(predictedSpecialNumber);
        }
    }
}
