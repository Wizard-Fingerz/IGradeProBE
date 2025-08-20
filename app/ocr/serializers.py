from rest_framework import serializers
from .models import StudentScript, ScriptPage

class ScriptPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScriptPage
        fields = ["page_number", "image", "extracted_text", "extracted_answers"]  # Include the new field

class StudentScriptSerializer(serializers.ModelSerializer):
    pages = ScriptPageSerializer(many=True, read_only=True)
 
    class Meta:
        model = StudentScript
        fields = ["id", "subject", 'subject_name', "uploaded_at", "student_id", "student_name", "candidate_number", "pages"]