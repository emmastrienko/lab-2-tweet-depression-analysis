namespace Lab2Sentiment.Models;

// Дані для навчання: очищений текст і мітка 0/1 -> bool
public class TweetData
{
    public string Text { get; set; }
    public bool Label { get; set; }
}
