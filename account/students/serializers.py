from account.serializers import UserSerializer
from app.subjects.models import StudentSubjectRegistration
from .models import *
from rest_framework import serializers, viewsets
from .models import User
from django.contrib.auth.hashers import make_password


class DetailedStudentSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = Student
        fields = '__all__'
        read_only_fields = ('username', 'password')



class StudentSerializer(serializers.ModelSerializer):
 
    class Meta:
        model = Student
        fields = '__all__'
        read_only_fields = ('username', 'password')

