import os
from typing import List

import nltk
import pandas as pd
import spacy
from Levenshtein import distance as lev_distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


class ParameterFinder:
    """
    A class to process text and extract parameters from the text based on a predefined list of parameters.
    """
    def __init__(self, params_csv="params.csv"):
        """
        Initializes the ParameterFinder with a list of parameters and necessary NLP tools.

        :param params_csv: Path to the CSV file containing the parameter list (default is "params.csv")
        """
        # Initialize necessary components
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Load parameter list and action enrichment dataset
        params_csv_path = os.path.abspath(os.path.join(os.getcwd(), f"data/{params_csv}"))
        self.params_df = pd.read_csv(params_csv_path)
        self.parameter_list = self.params_df["parameter"].dropna().astype(str).str.lower().tolist()

        # Create a list of parameter words for stop word filtering
        parameter_words = set(word for param in self.parameter_list for word in param.split())
        self.stop_words = [word for word in self.stop_words if word not in parameter_words]

    def _lemmatize_text(self, text: str) -> str:
        """
        Lemmatizes the given text, removing stopwords and performing other preprocessing steps.
        Named entities are skipped during lemmatization.

        :param text: The input text (sentence) to be lemmatized.
        :return: The preprocessed and lemmatized text.
        """
        doc = self.nlp(text.lower())
        named_entities = {ent.text for ent in doc.ents}
        words = word_tokenize(text.lower())

        # Remove stop words, symbols, and non-alphabetic characters
        words = [word for word in words if word.isalpha() and word not in self.stop_words]

        # Lemmatize the words, skipping named entities
        lemmatized_words = [
            self._lemmatize_word(word) if word not in named_entities else word
            for word in words
        ]
        result = " ".join(lemmatized_words)
        result = result.replace("tipped in", "tip in")
        result = result.replace("eurosteps", "euro step")
        return result

    def _lemmatize_word(self, word: str) -> str:
        """
        Lemmatizes a word based on its POS (part-of-speech) tag.

        :param word: The word to be lemmatized.
        :return: The lemmatized version of the word.
        """
        pos_tag = nltk.pos_tag([word])[0][1]
        if pos_tag.startswith('VB'):
            wordnet_pos = 'v'  # Verb
        elif pos_tag.startswith('NN'):
            wordnet_pos = 'n'  # Noun
        elif pos_tag.startswith('JJ'):
            wordnet_pos = 'a'  # Adjective
        else:
            wordnet_pos = 'n'  # Default to noun if unsure

        return self.lemmatizer.lemmatize(word, pos=wordnet_pos)

    def _detect_parameters_ngrams(self, sentence_bigrams: List[str], parameter_bigrams: List[str],
                                  max_distance: int = 1) -> List[str]:
        """
        Detects parameters from sentence bigrams that match the parameter bigrams using Levenshtein distance.

        :param sentence_bigrams: A list of bigrams (two-word sequences) from the sentence.
        :param parameter_bigrams: A list of bigrams (two-word sequences) from the parameter list.
        :param max_distance: The maximum allowed Levenshtein distance for a match (default is 1).
        :return: A list of matching parameter bigrams.
        """
        matches = []
        for sent_bigram in sentence_bigrams:
            for param_bigram in parameter_bigrams:
                lev_dist = lev_distance(sent_bigram, param_bigram)
                if lev_dist <= max_distance:
                    matches.append(param_bigram)
        return list(set(matches))

    def _get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Generates n-grams (sequences of n words) from the given text.

        :param text: The input text (sentence) to generate n-grams from.
        :param n: The size of the n-grams (default is 2 for bigrams).
        :return: A list of n-grams (as space-separated strings).
        """
        tokens = nltk.word_tokenize(text)
        n_grams = ngrams(tokens, n)
        return [' '.join(gram) for gram in n_grams]

    def _find_params(self, sentence: str) -> List[str]:
        """
        Finds parameters from the given sentence by matching the sentence to the predefined parameters.

        :param sentence: The input sentence in which parameters should be found.
        :return: A list of parameters found in the sentence.
        """
        found_params = [param for param in self.parameter_list if param in sentence]
        for n in reversed(range(1, 6)):
            if n > 1 or len(found_params) == 0:
                sentence_bigrams = self._get_ngrams(sentence, n=n)
                parameters_n_words = [param for param in self.parameter_list if len(param.split()) == n]
                if len(sentence_bigrams) > 0 and len(parameters_n_words) > 0:
                    found_params.extend(self._detect_parameters_ngrams(sentence_bigrams, parameters_n_words))

        if len(found_params) > 1 and 'jam' in found_params and 'jam' not in sentence.split(' '):
            found_params = [item for item in found_params if item != "jam"]
        if "fake" in found_params and "pump fake" in found_params:
            found_params = [item for item in found_params if item != "fake"]
        return list(set(found_params))

    def get_parameters(self, sentence: str) -> List[str]:
        """
        Main function to get parameters from the sentence after preprocessing and detecting the parameters.

        :param sentence: The input sentence (commentary text) to process.
        :return: A list of parameters found in the sentence.
        """
        processed_sentence = self._lemmatize_text(sentence.lower())
        found_params = self._find_params(processed_sentence)
        return found_params
