import nltk

# Descargar recursos necesarios de NLTK
try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

class CourseRecommender:
    def __init__(self):
        self.data = pd.read_csv('courses_dataset.csv')
        self.vectorizer = TfidfVectorizer()
        self._prepare_data()
        self.word2vec_model = self._train_word2vec()

    def _prepare_data(self):
        self.data.drop_duplicates(inplace=True)
        self.data.fillna('', inplace=True)
        en_stopwords = stopwords.words("english")
        lemma = WordNetLemmatizer()

        def clean(text):
            text = re.sub("[^A-Za-z1-9 ]", "", text)
            text = text.lower()
            tokens = word_tokenize(text)
            clean_list = [lemma.lemmatize(token) for token in tokens if token not in en_stopwords]
            return " ".join(clean_list)

        self.data['Description'] = self.data['title'] + ' ' + self.data['course_description']
        self.data['Description'] = self.data['Description'].apply(clean)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data['Description'])

    def _train_word2vec(self):
        sentences = [text.split() for text in self.data['Description']]
        return Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def get_word2vec_vector(self, word):
        try:
            return self.word2vec_model.wv[word]
        except KeyError:
            return np.zeros((100,))

    def get_combined_vector(self, text):
        words = text.split()
        tfidf_vector = self.vectorizer.transform([text]).toarray()[0]
        word2vec_vectors = np.array([self.get_word2vec_vector(word) for word in words])

        # Verificar si la suma de los pesos TF-IDF es cero
        if np.sum(tfidf_vector[:len(word2vec_vectors)]) == 0:
            # Retornar un vector de ceros o manejar el caso según sea necesario
            return np.zeros(word2vec_vectors.shape[1]) if len(word2vec_vectors) > 0 else np.zeros(self.word2vec_model.vector_size)

        # Promediar los vectores Word2Vec ponderados por TF-IDF
        weighted_word2vec_vector = np.average(word2vec_vectors, axis=0, weights=tfidf_vector[:len(word2vec_vectors)])
        
        return weighted_word2vec_vector

    def recommend_courses(self, course_index, num_recommendations=5):
        self.data['combined_vector'] = self.data['Description'].apply(self.get_combined_vector)
        cosine_similarities = cosine_similarity([self.data.iloc[course_index]['combined_vector']], self.data['combined_vector'].tolist())[0]
        similar_courses = cosine_similarities.argsort()[-num_recommendations-1:-1][::-1]
        return self.data.iloc[similar_courses]

    def get_courses(self, top_n=10):
        courses = []
        values = self.data.values.tolist()
        for item in values[:top_n]:
            iloc = self.data.iloc[item[0]]
            course = {"course_name": iloc["course_name"], "course_description": iloc["course_description"], "id": int(iloc["id"])}
            courses.append(course)
        return courses

    def get_course_recommendations(self, course_id):
        similarity = cosine_similarity(self.tfidf_matrix)
        row_num = self.data.index[self.data['id'] == course_id].tolist()
        if len(row_num) == 0:
            return []
        similar_courses = list(enumerate(similarity[row_num[0]]))
        sorted_similar_courses = sorted(similar_courses, key=lambda x: x[1], reverse=True)[:6]
        recommendations = []
        for item in sorted_similar_courses:
            iloc = self.data.iloc[item[0]]
            course = {"course_name": iloc["course_name"], "course_description": iloc["course_description"], "id": int(iloc["id"])}
            recommendations.append(course)
        return recommendations

    def get_search_recommendations(self, query, top_n=5):
        query_vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        similar_courses = list(enumerate(similarity))
        sorted_similar_courses = sorted(similar_courses, key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = []
        for item in sorted_similar_courses:
            iloc = self.data.iloc[item[0]]
            course = {"course_name": iloc["course_name"], "course_description": iloc["course_description"], "id": int(iloc["id"])}
            recommendations.append(course)
        return recommendations
