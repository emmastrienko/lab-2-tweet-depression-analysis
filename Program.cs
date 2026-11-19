// Лабораторна робота 2. Побудова класифікаційних моделей
// Варіант 5. Сентиментний аналіз: чи є у людини депресія (бінарна класифікація)
// Датасет: https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets
// Файл: sentiment_tweets3.csv  (Index, message to examine, label (depression result))

using Lab2Sentiment.Data;
using Lab2Sentiment.Evaluation;
using Lab2Sentiment.Models;
using Lab2Sentiment.Reporting;
using Lab2Sentiment.Training;
using Microsoft.ML;

var dataPath = args.Length > 0 ? args[0] : Path.Combine(AppContext.BaseDirectory, "data", "sentiment_tweets3.csv");
if (!File.Exists(dataPath))
{
    Console.WriteLine($"Файл не знайдено: {dataPath}");
    return;
}

var ml = new MLContext(seed: 42);

// 1. Завантаження даних
var cleaned = new TweetDataLoader(ml).Load(dataPath);
ConsoleReporter.PrintClassBalance(cleaned);

var data = ml.Data.LoadFromEnumerable(cleaned);
var split = ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 42);

// 2-3. Конвеєр ознак і набір тренерів
var featurizer = SentimentPipelineBuilder.BuildFeaturizer(ml);
var trainers = SentimentPipelineBuilder.GetTrainers(ml);
var evaluator = new ModelEvaluator(ml);

ConsoleReporter.PrintComparisonHeader();

var results = new List<TrainerRun>();
foreach (var option in trainers)
{
    var run = evaluator.TrainAndEvaluate(option, featurizer, split.TrainSet, split.TestSet);
    results.Add(run);
    ConsoleReporter.PrintComparisonRow(run);
}

// 4. Найкраща модель за F1 (датасет незбалансований, тому не за Accuracy)
var best = results.OrderByDescending(r => r.Metrics.F1Score).First();
ConsoleReporter.PrintBest(best);

// 5. Кросвалідація (5 фолдів) найкращої моделі на всьому наборі
var cvMetrics = evaluator.CrossValidate(best, data, numberOfFolds: 5);
ConsoleReporter.PrintCrossValidation(best.Name, cvMetrics);

// 6. Збереження моделі
var modelPath = Path.Combine(AppContext.BaseDirectory, "depression_model.zip");
ml.Model.Save(best.Model, split.TrainSet.Schema, modelPath);
Console.WriteLine($"Модель збережено: {modelPath}\n");

// 7. Перевірка на власних прикладах
var engine = ml.Model.CreatePredictionEngine<TweetData, TweetPrediction>(best.Model);
string[] samples =
{
    "I feel so empty and hopeless, nothing matters anymore",
    "Had an amazing day at the beach with my friends!",
    "I can't sleep again, I'm tired of everything and everyone",
    "Just finished my project, really proud of myself today",
    "My depression is getting worse and I don't know who to talk to"
};
ConsoleReporter.PrintSamplePredictions(engine, samples);
