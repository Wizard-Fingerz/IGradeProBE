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
        ref_name = 'ExamSubSubQuestionSerializer'  # Add ref_name to avoid conflict


class SubQuestionSerializer(serializers.ModelSerializer):
    """Serializer for sub-questions, including sub-sub-questions."""
    sub_questions = SubSubQuestionSerializer(many=True, required=False)

    class Meta:
        model = SubjectQuestion
        fields = ['id', 'question_number', 'comprehension', 'question',
                  'examiner_answer', 'question_score', 'is_optional', 'parent_question', 'sub_questions']
        ref_name = 'ExamSubQuestionSerializer'  # Add ref_name to avoid conflict


class SubjectQuestionNestedSerializer(serializers.ModelSerializer):
    """Serializer for main questions, including sub-questions."""
    sub_questions = SubQuestionSerializer(many=True, required=False)

    class Meta:
        model = SubjectQuestion
        fields = ['question_number', 'comprehension', 'question',
                  'examiner_answer', 'question_score', 'is_optional', 'sub_questions']
        ref_name = 'QuestionSubQuestionSerializer'

    def create(self, validated_data):
        sub_questions_data = validated_data.pop('sub_questions', [])
        subject = validated_data.get('subject')
        subject_question = SubjectQuestion.objects.create(**validated_data)

        for sub_question_data in sub_questions_data:
            sub_sub_questions_data = sub_question_data.pop('sub_questions', [])
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
        instance.exam_year = validated_data.get('exam_year', instance.exam_year)
        instance.exam_type = validated_data.get('exam_type', instance.exam_type)
        instance.total_mark = validated_data.get('total_mark', instance.total_mark)
        instance.paper_number = validated_data.get('paper_number', instance.paper_number)
        instance.subject = validated_data.get('subject', instance.subject)
        instance.save()

        # Track existing question IDs to avoid recreating them
        existing_question_ids = [question.id for question in instance.questions.all()]

        for question_data in questions_data:
            question_id = question_data.get('id', None)
            if question_id and question_id in existing_question_ids:
                # Update existing question
                question_instance = SubjectQuestion.objects.get(id=question_id)
                for attr, value in question_data.items():
                    if attr == 'sub_questions':  # Handle sub-questions separately
                        sub_questions_data = value
                        existing_sub_question_ids = [sub.id for sub in question_instance.sub_questions.all()]
                        for sub_question_data in sub_questions_data:
                            sub_question_id = sub_question_data.get('id', None)
                            if sub_question_id and sub_question_id in existing_sub_question_ids:
                                # Update existing sub-question
                                sub_question_instance = SubjectQuestion.objects.get(id=sub_question_id)
                                for sub_attr, sub_value in sub_question_data.items():
                                    setattr(sub_question_instance, sub_attr, sub_value)
                                sub_question_instance.save()
                            else:
                                # Create new sub-question
                                sub_question_data['parent_question'] = question_instance
                                SubjectQuestion.objects.create(**sub_question_data)
                    else:
                        setattr(question_instance, attr, value)
                question_instance.save()
            else:
                # Create new question using the create method
                sub_questions_data = question_data.pop('sub_questions', [])
                question_data['subject'] = instance.subject
                question_instance = SubjectQuestion.objects.create(**question_data)

                # Handle sub-questions for the new question
                for sub_question_data in sub_questions_data:
                    sub_question_data['parent_question'] = question_instance
                    SubjectQuestion.objects.create(**sub_question_data)

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

