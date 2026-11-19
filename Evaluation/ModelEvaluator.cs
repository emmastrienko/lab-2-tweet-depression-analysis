using Lab2Sentiment.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Lab2Sentiment.Evaluation;

// Навчання, оцінювання та кросвалідація бінарних класифікаторів
public class ModelEvaluator
{
    private readonly MLContext _ml;

    public ModelEvaluator(MLContext ml)
    {
        _ml = ml;
    }

    public TrainerRun TrainAndEvaluate(TrainerOption option, IEstimator<ITransformer> featurizer, IDataView trainSet, IDataView testSet)
    {
        var pipeline = featurizer.Append(option.Trainer);
        var model = pipeline.Fit(trainSet);
        var predictions = model.Transform(testSet);

        BinaryClassificationMetrics metrics;
        double? logLoss;
        if (option.Calibrated)
        {
            var calibratedMetrics = _ml.BinaryClassification.Evaluate(predictions);
            metrics = calibratedMetrics;
            logLoss = calibratedMetrics.LogLoss;
        }
        else
        {
            metrics = _ml.BinaryClassification.EvaluateNonCalibrated(predictions);
            logLoss = null;
        }

        return new TrainerRun(option.Name, pipeline, option.Calibrated, model, metrics, logLoss);
    }

    // Кросвалідація найкращої моделі на всьому наборі
    public List<BinaryClassificationMetrics> CrossValidate(TrainerRun run, IDataView data, int numberOfFolds)
    {
        var cvMetrics = run.Calibrated
            ? _ml.BinaryClassification.CrossValidate(data, run.Pipeline, numberOfFolds: numberOfFolds)
                .Select(f => (BinaryClassificationMetrics)f.Metrics)
            : _ml.BinaryClassification.CrossValidateNonCalibrated(data, run.Pipeline, numberOfFolds: numberOfFolds)
                .Select(f => f.Metrics);

        return cvMetrics.ToList();
    }
}
