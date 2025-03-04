from rest_framework import serializers
from .models import StudentScript, ScriptPage

class ScriptPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScriptPage
        fields = ["page_number", "image", "extracted_text", "extracted_answers"]  # Include the new field

class StudentScriptSerializer(serializers.ModelSerializer):
    pages = ScriptPageSerializer(many=True, read_only=True)
    student = serializers.StringRelatedField()  # Use StringRelatedField to display student information
    subject = serializers.StringRelatedField()  # Use StringRelatedField to display subject information

    class Meta:
        model = StudentScript
        fields = ["id", "subject", "uploaded_at", "student", "pages"]