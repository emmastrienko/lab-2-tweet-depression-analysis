# Лаб. 2, варіант 5: сентиментний аналіз (депресія / ні)

1. Завантажити датасет: https://www.kaggle.com/datasets/gargmanas/sentimental-analysis-for-tweets
2. Покласти `sentiment_tweets3.csv` у папку `data/`.
3. Запустити (потрібен .NET 8 SDK):
   ```
   dotnet run
   ```
   або зі своїм шляхом: `dotnet run -- path/to/file.csv`

Що робить програма:
- очищує твіти (посилання, @згадки, #, HTML-сутності);
- `FeaturizeText`: нижній регістр, стоп-слова, word 1–2-grams, char 3-grams;
- порівнює 6 тренерів (SdcaLogisticRegression, LbfgsLogisticRegression, SgdCalibrated, AveragedPerceptron, LinearSvm, FastTree) на тестовій вибірці 80/20: Accuracy, AUC, Precision, Recall, F1, LogLoss;
- обирає найкращу модель за F1, бо класи незбалансовані, тому Accuracy тут вводить в оману;
- робить кросвалідацію (5 фолдів) для найкращої моделі: `CrossValidate` для каліброваних тренерів і `CrossValidateNonCalibrated` для некаліброваних;
- зберігає модель у `depression_model.zip` і робить прогнози на власних прикладах.
`