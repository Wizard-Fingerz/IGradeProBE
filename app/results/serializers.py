
from rest_framework import serializers

from app.results.models import ExamResult


class ExamResultSerializer(serializers.ModelSerializer):
    similarity_score_percentage = serializers.SerializerMethodField()

    class Meta:
        model = ExamResult
        fields = '__all__'

    def get_similarity_score_percentage(self, obj):
        if obj.similarity_score is not None:
            return f"{obj.similarity_score * 100:.0f}%"
        return None
