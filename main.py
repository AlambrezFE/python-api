from flask import Flask, request, jsonify
from tfidfmodel import CourseRecommender

app = Flask(__name__)
recommender = CourseRecommender()

@app.route('/api/course', methods=['GET'])
def courses_index():
    limit = request.args.get('limit')
    limit = int(limit if limit is not None else 10)
    recommendations = recommender.get_courses(limit)
    return jsonify(recommendations)

@app.route('/api/course/<int:course_id>', methods=['GET'])
def recommend_course(course_id):
    recommendations = recommender.get_course_recommendations(course_id)
    return jsonify(recommendations)

@app.route('/api/search', methods=['GET'])
def search_courses():
    query = request.args.get('query')
    limit = request.args.get('limit')
    limit = int(limit if limit is not None else 5)
    recommendations = recommender.get_search_recommendations(query, limit)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
