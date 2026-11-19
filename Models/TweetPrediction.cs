using Microsoft.ML.Data;

namespace Lab2Sentiment.Models;

// Вихід моделі
public class TweetPrediction
{
    [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
    public float Score { get; set; }
}
