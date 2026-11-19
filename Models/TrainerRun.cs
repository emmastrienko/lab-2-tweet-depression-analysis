using Microsoft.ML;
using Microsoft.ML.Data;

namespace Lab2Sentiment.Models;

// Результат навчання та оцінювання одного тренера (LogLoss є лише в каліброваних моделей)
public sealed record TrainerRun(
    string Name, IEstimator<ITransformer> Pipeline, bool Calibrated, ITransformer Model,
    BinaryClassificationMetrics Metrics, double? LogLoss);
