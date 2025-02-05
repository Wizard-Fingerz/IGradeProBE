from django.db import models
from django.db.models import Sum

from account.models import User
from account.students.models import Student




class Subject(models.Model):
    examiner = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='examiner_subject', null=True, blank=True)
    name = models.CharField(max_length=250)
    code = models.CharField(max_length=250)
    description = models.CharField(max_length=250)
    
    
    

    def __str__(self):
        return str(self.name) or ''



class StudentSubjectRegistration(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    registration_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        # Enforce unique constraint on student and Subject combination
        unique_together = ['student', 'subject']

    def __str__(self):
        return str(self.student.user.username) or ''
