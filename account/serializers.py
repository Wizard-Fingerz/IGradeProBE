from .models import *
from rest_framework import serializers, viewsets
from .models import User
from django.contrib.auth.hashers import make_password

class UserSerializer(serializers.ModelSerializer):
    student_id = serializers.CharField(max_length=15, required=False)
    examiner_id = serializers.CharField(max_length=15, required=False)

    class Meta:
        model = User
        fields = '__all__'

    def validate_student_id(self, value):
        # Your custom validation for matric number here, if needed
        return value

    def validate_examiner_id(self, value):
        # Your custom validation for examiner id here, if needed
        return value

    def create(self, validated_data):
        student_id = validated_data.pop('student_id', None)
        examiner_id = validated_data.pop('examiner_id', None)

        if student_id:
            username = student_id.replace('/', '/')
            existing_user = User.objects.filter(student_id=student_id).first()

        elif examiner_id:
            username = examiner_id.replace('/', '/')
            existing_user = User.objects.filter(examiner_id=examiner_id).first()

        if existing_user:
            for key, value in validated_data.items():
                setattr(existing_user, key, value)
            existing_user.save()
            return existing_user
        else:
            user = User.objects.create_user(username=username, **validated_data)
            return user


