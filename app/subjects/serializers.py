
from rest_framework import serializers
from app.subjects.models import Subject


class SubjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subject
        fields = ['id', 'examiner', 'name', 'code', 'description']
