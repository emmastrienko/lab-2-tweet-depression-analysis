using Microsoft.ML;

namespace Lab2Sentiment.Models;

// Опис одного тренера: чи повертає модель ймовірність (Calibrated)
public sealed record TrainerOption(string Name, IEstimator<ITransformer> Trainer, bool Calibrated);
