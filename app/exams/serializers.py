from rest_framework import serializers

from app.questions.models import SubjectQuestion
from .models import Exam


class ExamSerializer(serializers.ModelSerializer):

    class Meta:
        model = Exam
        fields = [
            'id', 'created_by', 'questions', 'duration', 'instruction',
            'subject', 'total_mark', 'exam_type', 'exam_year', 'question_count', 'subject_name', 'created_by_name',
            'paper_number', 'is_activate', 'subject_code',
        ]
        read_only_fields = ['id']


class SubQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectQuestion
        fields = ['id', 'question_number', 'comprehension', 'question',
                  'examiner_answer', 'question_score', 'is_optional', 'parent_question']


class SubjectQuestionNestedSerializer(serializers.ModelSerializer):
    sub_questions = SubQuestionSerializer(many=True, required=False)

    class Meta:
        model = SubjectQuestion
        fields = ['question_number', 'comprehension', 'question',
                  'examiner_answer', 'question_score', 'is_optional', 'sub_questions']

    def create(self, validated_data):
        sub_questions_data = validated_data.pop('sub_questions', [])
        subject = validated_data.get('subject')
        subject_question = SubjectQuestion.objects.create(**validated_data)

        for sub_question_data in sub_questions_data:
            sub_question_data['subject'] = subject
            sub_question_data['parent_question'] = subject_question
            SubjectQuestion.objects.create(**sub_question_data)

        return subject_question
    


class CreateExamSerializer(serializers.ModelSerializer):
    questions = SubjectQuestionNestedSerializer(many=True)

    class Meta:
        model = Exam
        fields = ['exam_year', 'exam_type', 'total_mark',
                  'paper_number', 'subject', 'questions']

    def create(self, validated_data):
        questions_data = validated_data.pop('questions', [])
        exam = Exam.objects.create(**validated_data)

        for question_data in questions_data:
            question_data['subject'] = exam.subject
            question_instance = SubjectQuestionNestedSerializer().create(question_data)
            exam.questions.add(question_instance)

        return exam

class GetExamDetailSerializer(serializers.ModelSerializer):
    questions = SubjectQuestionNestedSerializer(many=True, read_only=True)  # Ensure full question details

    class Meta:
        model = Exam
        
        fields = ['exam_year', 'exam_type', 'total_mark',
                  'paper_number', 'subject', 'questions']
        read_only_fields = ['id']

