import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


class EDA:
    def __init__(self, transcript_file="action_enrichment_ds_home_exercise.csv", params_file="params.csv"):
        """Initialize with dataset and parameter file containing possible actions."""
        self.df = pd.read_csv(transcript_file)
        self.params = pd.read_csv(params_file)['parameter'].dropna().astype(str).str.lower().tolist()

    def overview(self):
        """Print basic dataset information."""
        print("\n--- Dataset Overview ---")
        print(self.df.info())
        print("\nFirst 5 rows:")
        print(self.df.head())
        print("\nDataset Shape:", self.df.shape)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        print("\nDuplicate Rows:", self.df.duplicated().sum())

    def label_distribution(self):
        """Plot distribution of labels (valid/invalid)."""
        plt.figure(figsize=(6, 4))
        ax = sns.countplot(x=self.df['Label'], palette='coolwarm')
        plt.title('Label Distribution')
        plt.xlabel('Valid (1) vs. Invalid (0)')
        plt.ylabel('Count')
        plt.savefig('label_distribution.png', bbox_inches='tight')
        plt.show()

        # Printing label distribution data
        label_counts = self.df['Label'].value_counts(normalize=True)
        print("\n--- Label Distribution Data ---")
        print(label_counts)

    def text_length_distribution(self):
        """Analyze distribution of text lengths."""
        self.df['text_length'] = self.df['Text'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(8, 5))
        ax = sns.histplot(self.df['text_length'], bins=20, kde=True, color='blue')
        plt.title('Text Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.savefig('text_length_distribution.png', bbox_inches='tight')
        plt.show()

        # Printing text length distribution data
        print("\n--- Text Length Distribution Data ---")
        print(self.df['text_length'].describe())

    def keyword_analysis(self):
        """Find most common action phrases from params.csv in the text dataset."""
        action_counts = Counter()
        for text in self.df['Text']:
            for action in self.params:
                if action in text.lower():
                    action_counts[action] += 1

        action_df = pd.DataFrame(action_counts.items(), columns=['Action Phrase', 'Count'])
        action_df = action_df.sort_values(by='Count', ascending=False)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(y=action_df['Action Phrase'][:15], x=action_df['Count'][:15], palette='viridis')
        plt.title('Most Common Action Phrases')
        plt.xlabel('Frequency')
        plt.ylabel('Action Phrase')
        plt.savefig('most_common_action_phrases.png', bbox_inches='tight')
        plt.show()

        # Printing the most common action phrases data
        print("\n--- Most Common Action Phrases ---")
        print(action_df.head(15))

    def word_cloud(self):
        """Generate a word cloud from text data."""
        text_corpus = ' '.join(self.df['Text'])
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text_corpus)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Commentary Text")
        plt.savefig('word_cloud.png', bbox_inches='tight')
        plt.show()

    def bigram_trigram_analysis(self, ngram_range=(2, 3), top_n=10):
        """Analyze most common bigrams and trigrams."""
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        ngram_matrix = vectorizer.fit_transform(self.df['Text'])
        ngram_counts = np.array(ngram_matrix.sum(axis=0))[0]
        ngram_freq = dict(zip(vectorizer.get_feature_names_out(), ngram_counts))

        # Sort and plot the most frequent n-grams
        ngram_df = pd.DataFrame(sorted(ngram_freq.items(), key=lambda x: x[1], reverse=True),
                                columns=['N-gram', 'Count']).head(top_n)

        plt.figure(figsize=(10, 5))
        sns.barplot(y=ngram_df['N-gram'], x=ngram_df['Count'], palette='plasma')
        plt.title(f"Top {top_n} Most Frequent {ngram_range} n-grams")
        plt.xlabel("Frequency")
        plt.ylabel("N-gram")
        plt.savefig(f'top_{top_n}_ngram_analysis.png', bbox_inches='tight')
        plt.show()

        # Printing the n-gram analysis data
        print(f"\n--- Top {top_n} Most Frequent {ngram_range} n-grams ---")
        print(ngram_df)

    def action_distribution(self):
        """Plot distribution of action types in the dataset."""
        action_counts = self.df['action_1'].value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(y=action_counts.index, x=action_counts.values, palette='coolwarm')
        plt.title('Action Type Distribution')
        plt.xlabel('Number of Examples')
        plt.ylabel('Action Type')
        plt.savefig('action_type_distribution.png', bbox_inches='tight')
        plt.show()

        # Printing action distribution data
        print("\n--- Action Type Distribution Data ---")
        print(action_counts)

    def action_validity(self):
        """Calculate probability of each action being valid (Label=1) and visualize it."""
        action_stats = self.df.groupby('action_1')['Label'].value_counts().unstack(fill_value=0)
        action_stats['Total'] = action_stats[0] + action_stats[1]
        action_stats['Valid Probability'] = action_stats[1] / action_stats['Total']

        action_stats = action_stats.sort_values(by='Valid Probability', ascending=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(y=action_stats.index, x=action_stats['Valid Probability'], palette='viridis')
        plt.title('Probability of Action Being Valid')
        plt.xlabel('Probability of Valid Label (Label = 1)')
        plt.ylabel('Action Type')
        plt.xlim(0, 1)
        plt.savefig('action_validity_probability.png', bbox_inches='tight')
        plt.show()

        # Printing action validity data
        print("\n--- Action Validity Probability Data ---")
        print(action_stats[['Valid Probability', 'Total']])

    def sentiment_analysis(self):
        """Perform sentiment analysis and visualize it."""
        self.df['Sentiment'] = self.df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

        plt.figure(figsize=(8, 5))
        sns.histplot(self.df['Sentiment'], bins=30, kde=True, color='purple')
        plt.title('Sentiment Analysis of Text')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.savefig('sentiment_analysis.png', bbox_inches='tight')
        plt.show()

        # Printing sentiment analysis data
        print("\n--- Sentiment Analysis Data ---")
        print(self.df['Sentiment'].describe())

    def action_phrase_invalid_analysis(self):
        """Analyze the relationship between action phrases and invalid labels."""
        invalid_action_df = self.df[self.df['Label'] == 0]
        invalid_action_counts = Counter()

        for text in invalid_action_df['Text']:
            for action in self.params:
                if action in text.lower():
                    invalid_action_counts[action] += 1

        invalid_action_df = pd.DataFrame(invalid_action_counts.items(), columns=['Action Phrase', 'Invalid Count'])
        invalid_action_df = invalid_action_df.sort_values(by='Invalid Count', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(y=invalid_action_df['Action Phrase'][:15], x=invalid_action_df['Invalid Count'][:15],
                    palette='magma')
        plt.title('Most Common Invalid Action Phrases')
        plt.xlabel('Invalid Count')
        plt.ylabel('Action Phrase')
        plt.savefig('invalid_action_phrases.png', bbox_inches='tight')
        plt.show()

        # Printing invalid action phrases data
        print("\n--- Invalid Action Phrases Data ---")
        print(invalid_action_df.head(15))

    def text_similarity(self):
        """Compute cosine similarity between text data."""
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['Text'])
        cosine_sim = cosine_similarity(tfidf_matrix)

        plt.figure(figsize=(8, 5))
        sns.heatmap(cosine_sim, cmap='coolwarm', xticklabels=self.df['Text'].head(10),
                    yticklabels=self.df['Text'].head(10))
        plt.title('Cosine Similarity Heatmap')
        plt.savefig('cosine_similarity_heatmap.png', bbox_inches='tight')
        plt.show()

    def text_length_vs_validity(self):
        """Analyze correlation between text length and validity."""
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='Label', y='text_length', data=self.df, palette='coolwarm')
        plt.title('Text Length vs Validity')
        plt.xlabel('Validity (1: Valid, 0: Invalid)')
        plt.ylabel('Text Length (in words)')
        plt.savefig('text_length_vs_validity.png', bbox_inches='tight')
        plt.show()

    def word_cloud_by_label(self):
        """Generate word clouds for valid vs invalid actions."""
        valid_text = ' '.join(self.df[self.df['Label'] == 1]['Text'])
        invalid_text = ' '.join(self.df[self.df['Label'] == 0]['Text'])

        valid_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(valid_text)
        invalid_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(invalid_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(valid_wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud for Valid Actions")
        plt.savefig('valid_word_cloud.png', bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.imshow(invalid_wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud for Invalid Actions")
        plt.savefig('invalid_word_cloud.png', bbox_inches='tight')
        plt.show()

    def explore_all(self):
        """Run all EDA methods."""
        self.overview()
        self.label_distribution()
        self.text_length_distribution()
        self.keyword_analysis()
        self.word_cloud()
        self.bigram_trigram_analysis()
        self.action_distribution()
        self.action_validity()
        self.sentiment_analysis()
        self.action_phrase_invalid_analysis()
        # self.text_similarity()
        self.text_length_vs_validity()
        # self.word_cloud_by_label()


if __name__ == "__main__":
    eda = EDA()
    eda.explore_all()
