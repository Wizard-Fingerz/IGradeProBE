from django.db import models

from account.students.models import Student

class StudentScript(models.Model):
    student_id = models.ForeignKey(Student, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Script for Student {self.student_id}"

class ScriptPage(models.Model):
    script = models.ForeignKey(StudentScript, related_name="pages", on_delete=models.CASCADE)
    image = models.ImageField(upload_to="scripts/")
    extracted_text = models.TextField(blank=True, null=True)
