using System.Text.RegularExpressions;

namespace Lab2Sentiment.Data;

// Очищення тексту твітів: посилання, згадки, хештеги, HTML-мнемоніки, зайві пробіли
public static class TextCleaner
{
    public static string Clean(string text)
    {
        if (string.IsNullOrEmpty(text)) return string.Empty;
        text = Regex.Replace(text, @"https?://\S+|www\.\S+", " ");
        text = Regex.Replace(text, @"@\w+", " ");
        text = text.Replace("#", " ");
        text = Regex.Replace(text, @"&\w+;", " ");
        return Regex.Replace(text, @"\s+", " ").Trim();
    }
}
