from rest_framework import serializers




from app.questions.models import SubjectQuestion
from app.subjects.models import Subject
from app.subjects.serializers import SubjectSerializer


class GetSubjectQuestionSerializer(serializers.ModelSerializer):
    subject = SubjectSerializer()

    class Meta:
        model = SubjectQuestion
        fields = "__all__"



class SubQuestionSerializer(serializers.ModelSerializer):
    class Meta:
        model = SubjectQuestion
        fields = ['id', 'question_number', 'comprehension', 'question', 'examiner_answer', 'question_score', 'is_optional', 'parent_question']


class SubjectQuestionSerializer(serializers.ModelSerializer):
    parent_question = serializers.PrimaryKeyRelatedField(
        queryset=SubjectQuestion.objects.all(), required=False
    )
    sub_questions = SubQuestionSerializer(many=True, read_only=True)

    class Meta:
        model = SubjectQuestion
        fields = ['id', 'question_number', 'subject_code', 'subject_name', 'comprehension', 'question', 'examiner_answer', 'question_score', 'is_optional', 'parent_question', 'sub_questions', ]


class SubjectQuestionNestedSerializer(serializers.ModelSerializer):
    sub_questions = SubQuestionSerializer(many=True, required=False)

    class Meta:
        model = SubjectQuestion
        fields = ['question_number', 'comprehension', 'question', 'examiner_answer', 'question_score', 'is_optional', 'sub_questions', 'subject_name', 'subject_code']

    def create(self, validated_data):
        sub_questions_data = validated_data.pop('sub_questions', [])
        subject = validated_data.get('Subject')
        subject_question = SubjectQuestion.objects.create(**validated_data)
        
        for sub_question_data in sub_questions_data:
            sub_question_data['subject'] = Subject
            sub_question_data['parent_question'] = subject_question
            SubjectQuestion.objects.create(**sub_question_data)
        
        return subject_question

