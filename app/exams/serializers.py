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

class SubSubQuestionSerializer(serializers.ModelSerializer):
    """Serializer for sub-sub-questions."""
    class Meta:
        model = SubjectQuestion
        fields = ['id', 'question_number', 'comprehension', 'question',
                  'examiner_answer', 'question_score', 'is_optional', 'parent_question']
        ref_name = 'ExamSubSubQuestionSerializer'

class SubQuestionSerializer(serializers.ModelSerializer):
    """Serializer for sub-questions, including sub-sub-questions."""
    sub_sub_questions = SubSubQuestionSerializer(many=True, required=False)

    class Meta:
        model = SubjectQuestion
        fields = ['id', 'question_number', 'comprehension', 'question',
                  'examiner_answer', 'question_score', 'is_optional', 'parent_question', 'sub_sub_questions']
        ref_name = 'ExamSubQuestionSerializer'

class SubjectQuestionNestedSerializer(serializers.ModelSerializer):
    """Serializer for main questions, including sub-questions."""
    sub_questions = SubQuestionSerializer(many=True, required=False)

    class Meta:
        model = SubjectQuestion
        fields = ['id', 'question_number', 'comprehension', 'question',
                  'examiner_answer', 'question_score', 'is_optional', 'sub_questions']
        ref_name = 'QuestionSubQuestionSerializer'

    def create(self, validated_data):
        sub_questions_data = validated_data.pop('sub_questions', [])
        subject = validated_data.get('subject')
        subject_question = SubjectQuestion.objects.create(**validated_data)

        for sub_question_data in sub_questions_data:
            sub_sub_questions_data = sub_question_data.pop('sub_sub_questions', [])
            print(sub_sub_questions_data)
            sub_question_data['subject'] = subject
            sub_question_data['parent_question'] = subject_question
            sub_question_instance = SubjectQuestion.objects.create(**sub_question_data)

            # Handle sub-sub-questions
            for sub_sub_question_data in sub_sub_questions_data:
                sub_sub_question_data['subject'] = subject
                sub_sub_question_data['parent_question'] = sub_question_instance
                SubjectQuestion.objects.create(**sub_sub_question_data)

        return subject_question
    

class CreateExamSerializer(serializers.ModelSerializer):
    """Serializer for creating exams, including nested questions."""
    questions = SubjectQuestionNestedSerializer(many=True)

    class Meta:
        model = Exam
        fields = ['id', 'exam_year', 'exam_type', 'total_mark',
                  'paper_number', 'subject', 'questions']

    def create(self, validated_data):
        questions_data = validated_data.pop('questions', [])
        exam = Exam.objects.create(**validated_data)

        for question_data in questions_data:
            question_data['subject'] = exam.subject
            question_instance = SubjectQuestionNestedSerializer().create(question_data)
            exam.questions.add(question_instance)

        return exam
    
    
    def update(self, instance, validated_data):
        questions_data = validated_data.pop('questions', [])

        # Update exam fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        # Delete existing questions (cascade should handle sub-questions and sub-sub)
        instance.questions.all().delete()

        # Recreate questions
        for question_data in questions_data:
            question_data['subject'] = instance.subject
            question_instance = SubjectQuestionNestedSerializer().create(question_data)
            instance.questions.add(question_instance)

        return instance



class GetExamDetailSerializer(serializers.ModelSerializer):
    """Serializer for retrieving exam details, including nested questions."""
    questions = SubjectQuestionNestedSerializer(many=True, read_only=True)

    class Meta:
        model = Exam
        fields = ['exam_year', 'exam_type', 'total_mark',
                  'paper_number', 'subject', 'questions']
        read_only_fields = ['id']

