from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import nltk
nltk.downloader.download('vader_lexicon')
nltk.downloader.download('punkt')

def score_paragraph(para):
    sumeOfScores=0
    sid = SentimentIntensityAnalyzer()
    lines_list = tokenize.sent_tokenize(para)
    for sentence in lines_list:
        ss= sid.polarity_scores(sentence)
        sumeOfScores+=ss['compound']

    return sumeOfScores

