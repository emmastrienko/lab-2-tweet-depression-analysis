using Lab2Sentiment.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Lab2Sentiment.Training;

// Побудова конвеєра ознак і набору тренерів для порівняння
public static class SentimentPipelineBuilder
{
    // Ознаки з тексту: нижній регістр, стоп-слова, word 1-2 grams, char 3-grams, L2-нормування
    public static IEstimator<ITransformer> BuildFeaturizer(MLContext ml)
    {
        return ml.Transforms.Text.FeaturizeText("Features", new TextFeaturizingEstimator.Options
        {
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            KeepDiacritics = false,
            KeepPunctuations = false,
            KeepNumbers = false,
            StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options
            {
                Language = TextFeaturizingEstimator.Language.English
            },
            WordFeatureExtractor = new WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },
            CharFeatureExtractor = new WordBagEstimator.Options { NgramLength = 3, UseAllLengths = false },
            Norm = TextFeaturizingEstimator.NormFunction.L2
        }, nameof(TweetData.Text));
    }

    public static List<TrainerOption> GetTrainers(MLContext ml)
    {
        return new List<TrainerOption>
        {
            new("SdcaLogisticRegression", ml.BinaryClassification.Trainers.SdcaLogisticRegression(), true),
            new("LbfgsLogisticRegression", ml.BinaryClassification.Trainers.LbfgsLogisticRegression(), true),
            new("SgdCalibrated", ml.BinaryClassification.Trainers.SgdCalibrated(), true),
            new("AveragedPerceptron", ml.BinaryClassification.Trainers.AveragedPerceptron(numberOfIterations: 10), false),
            new("LinearSvm", ml.BinaryClassification.Trainers.LinearSvm(numberOfIterations: 20), false),
            new("FastTree", ml.BinaryClassification.Trainers.FastTree(numberOfLeaves: 20, numberOfTrees: 100), true),
        };
    }
}
