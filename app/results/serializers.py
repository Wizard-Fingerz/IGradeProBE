
from rest_framework import serializers

from app.results.models import ExamResult


class ExamResultSerializer(serializers.ModelSerializer):
    similarity_score_percentage = serializers.SerializerMethodField()

    class Meta:
        model = ExamResult
        fields = ['id', 'student', 'question', 'student_answer', 'student_score', 'similarity_score',
                  'similarity_score_percentage', 'attempted', 'question_text', 'question_number', 'question_score','examiner_answer','candidate_number']
        read_only_fields = ['similarity_score_percentage', 'question_text', 'question_number', 'question_score', 'examiner_answer','candidate_number']

    def get_similarity_score_percentage(self, obj):
        if obj.similarity_score is not None:
            return f"{obj.similarity_score * 100:.0f}%"
        return None
