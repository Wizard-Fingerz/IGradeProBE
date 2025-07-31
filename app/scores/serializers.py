from rest_framework import serializers

from account.students.serializers import DetailedStudentSerializer
from app.results.models import ExamResult
from app.scores.models import ExamResultScore


class ExamResultScoreSerializer(serializers.ModelSerializer):

    class Meta:
        model = ExamResultScore
        fields = ['id', 'exam_total_mark', 'student', 'exam_score', 'student_name', 'student_detials',
                  'grade', 'subject_detials', 'percentage_score', 'effective_total_marks']
        read_only_fields = ['grade', 'exam_total_mark', 'student_name', 'student_detials',
                            'subject_detials', 'percentage_score', 'effective_total_marks']


class AnswerScoreSerializer(serializers.ModelSerializer):
    student = serializers.CharField(
        source='student.user.username', read_only=True)
    question = serializers.CharField(
        source='question.question', read_only=True)
    question_number = serializers.CharField(
        source='question.question_number', read_only=True)
    Subject_name = serializers.CharField(
        source='question.Subject.title', read_only=True)
    Subject_code = serializers.CharField(
        source='question.Subject.code', read_only=True)
    similarity_score_percentage = serializers.SerializerMethodField()
    # question_score = serializers.CharField(source='Subject_question.question_score', read_only=True)
    question_score = serializers.CharField(
        source='question.question_score', read_only=True)

    class Meta:
        model = ExamResult
        fields = ['student', 'Subject_code', 'Subject_name', 'question_number',
                  'question', 'student_answer', 'student_score', 'question_score', 'similarity_score_percentage']

    def get_similarity_score_percentage(self, obj):
        if obj.similarity_score is not None:
            return f"{obj.similarity_score * 100:.0f}%"
        return None
