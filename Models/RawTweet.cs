using Microsoft.ML.Data;

namespace Lab2Sentiment.Models;

// Рядок CSV: Index, message to examine, label (depression result)
public class RawTweet
{
    [LoadColumn(1)] public string Text { get; set; }
    [LoadColumn(2)] public float Label { get; set; }
}
