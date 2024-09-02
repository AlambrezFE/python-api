import nltk
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    nltk.download('punkt_tab')

class CourseRecommender:
    def __init__(self):
        self.data = pd.read_csv('courses_dataset.csv')
        self.vectorizer = TfidfVectorizer()
        self._prepare_data()

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

    def get_courses(self, top_n=10):
        courses = []
        values = self.data.index.tolist()
        for idx in values[:top_n]:
            row = self.data.iloc[idx]
            course = {
                "course_name": row["title"], 
                "course_description": row["course_description"], 
                "id": int(row.get("id", idx))
            }
            courses.append(course)
        return courses

    def get_course_recommendations(self, course_id):
        similarity = cosine_similarity(self.tfidf_matrix)
        if 'id' not in self.data.columns:
            self.data['id'] = self.data.index  # Use index as ID if 'id' column doesn't exist
        row_num = self.data.index[self.data['id'] == course_id].tolist()
        if len(row_num) == 0:
            return []
        similar_courses = list(enumerate(similarity[row_num[0]]))
        sorted_similar_courses = sorted(similar_courses, key=lambda x: x[1], reverse=True)[:6]
        recommendations = []
        for item in sorted_similar_courses:
            iloc = self.data.iloc[item[0]]
            course = {
                "course_name": iloc["title"], 
                "course_description": iloc["course_description"], 
                "id": int(iloc.get("id", item[0]))
            }
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
            course = {
                "course_name": iloc["title"], 
                "course_description": iloc["course_description"], 
                "id": int(iloc.get("id", item[0]))
            }
            recommendations.append(course)
        return recommendations




