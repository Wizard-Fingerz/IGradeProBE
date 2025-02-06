from rest_framework import serializers
from .models import StudentScript, ScriptPage

class ScriptPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScriptPage
        fields = '__all__'

class StudentScriptSerializer(serializers.ModelSerializer):
    pages = ScriptPageSerializer(many=True, read_only=True)

    class Meta:
        model = StudentScript
        fields = '__all__'
