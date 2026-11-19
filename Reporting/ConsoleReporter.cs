using Lab2Sentiment.Data;
using Lab2Sentiment.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Lab2Sentiment.Reporting;

// Виведення статистики та результатів у консоль
public static class ConsoleReporter
{
    public static void PrintClassBalance(List<TweetData> data)
    {
        int pos = data.Count(t => t.Label);
        Console.WriteLine($"Записів: {data.Count}; депресія (1): {pos} ({100.0 * pos / data.Count:F1}%); " +
                          $"без депресії (0): {data.Count - pos} ({100.0 * (data.Count - pos) / data.Count:F1}%)\n");
    }

    public static void PrintComparisonHeader()
    {
        Console.WriteLine($"{"Тренер",-26}{"Accuracy",10}{"AUC",8}{"Precision",11}{"Recall",9}{"F1",8}{"LogLoss",10}");
        Console.WriteLine(new string('-', 82));
    }

    public static void PrintComparisonRow(TrainerRun run)
    {
        string logLoss = run.LogLoss.HasValue ? run.LogLoss.Value.ToString("F3") : "-";
        Console.WriteLine($"{run.Name,-26}{run.Metrics.Accuracy,10:P2}{run.Metrics.AreaUnderRocCurve,8:F3}{run.Metrics.PositivePrecision,11:P2}" +
                          $"{run.Metrics.PositiveRecall,9:P2}{run.Metrics.F1Score,8:F3}{logLoss,10}");
    }

    public static void PrintBest(TrainerRun best)
    {
        Console.WriteLine($"\nНайкраща модель за F1: {best.Name}");
        Console.WriteLine(best.Metrics.ConfusionMatrix.GetFormattedConfusionTable());
    }

    public static void PrintCrossValidation(string name, List<BinaryClassificationMetrics> cv)
    {
        Console.WriteLine($"Кросвалідація (5 фолдів) для {name}:");
        for (int i = 0; i < cv.Count; i++)
            Console.WriteLine($"  Фолд {i + 1}: Accuracy={cv[i].Accuracy:P2}  F1={cv[i].F1Score:F3}  AUC={cv[i].AreaUnderRocCurve:F3}");
        Console.WriteLine($"  Середнє: Accuracy={cv.Average(x => x.Accuracy):P2}  F1={cv.Average(x => x.F1Score):F3}  " +
                          $"AUC={cv.Average(x => x.AreaUnderRocCurve):F3}  (σ F1={StdDev(cv.Select(x => x.F1Score)):F3})\n");
    }

    public static void PrintSamplePredictions(PredictionEngine<TweetData, TweetPrediction> engine, string[] samples)
    {
        Console.WriteLine("Прогнози для власних прикладів:");
        foreach (var s in samples)
        {
            var p = engine.Predict(new TweetData { Text = TextCleaner.Clean(s) });
            Console.WriteLine($"  [{(p.PredictedLabel ? "ДЕПРЕСІЯ" : "норма   ")}] score={p.Score,7:F3}  {s}");
        }
    }

    private static double StdDev(IEnumerable<double> values)
    {
        var v = values.ToList();
        var mean = v.Average();
        return Math.Sqrt(v.Sum(x => (x - mean) * (x - mean)) / v.Count);
    }
}
