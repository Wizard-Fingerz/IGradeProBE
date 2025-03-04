from django.db import models

from account.students.models import Student
from app.subjects.models import Subject

class StudentScript(models.Model):
    student_id = models.ForeignKey(Student, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    @property
    def subject_name(self):
        return self.subject.name
    
    @property
    def student_name(self):
        return self.student_id.get_full_name()

    def __str__(self):
        return f"Script for Student {self.student_id}"

class ScriptPage(models.Model):
    script = models.ForeignKey(StudentScript, related_name="pages", on_delete=models.CASCADE)
    image = models.ImageField(upload_to="scripts/")
    extracted_text = models.TextField(blank=True, null=True)
    page_number = models.IntegerField(blank=True, null=True)  # Corrected typo
    extracted_answers = models.TextField(blank=True, null=True)  # Add a field for extracted answers