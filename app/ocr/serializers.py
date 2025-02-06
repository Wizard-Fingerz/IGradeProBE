from rest_framework import serializers
from .models import StudentScript, ScriptPage

class ScriptPageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScriptPage
        fields = ["page_number", "image", "extracted_text"]

class StudentScriptSerializer(serializers.ModelSerializer):
    pages = ScriptPageSerializer(many=True)

    class Meta:
        model = StudentScript
        fields = ["id", "uploaded_at", "student_id", "pages"]
