
from django.db import models
from account.models import User
from app.questions.models import SubjectQuestion
from app.subjects.models import Subject


class Exam(models.Model):
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name='exam_creator', null=True, blank=True)
    questions = models.ManyToManyField(
        SubjectQuestion, related_name='questions')
    duration = models.DurationField(blank= True, null = True)
    instruction = models.CharField(max_length=250, null = True, blank = True)
    subject = models.OneToOneField(Subject, on_delete=models.CASCADE, null = True, blank = True)
    total_mark = models.IntegerField()
    exam_type = models.CharField(max_length=150, blank = True,null = True)
    exam_year = models.IntegerField(blank = True, null = True)
    paper_number = models.IntegerField(blank = True, null = True)
    is_activate = models.BooleanField(default = False)

    @property
    def question_count(self):
        return self.questions.count()
    
    @property
    def subject_name(self):
        return self.subject.name
    
    @property
    def subject_code(self):
        return self.subject.code
    
    @property
    def created_by_name(self):
        return self.created_by.get_full_name()

    def __str__(self):
        return str(self.subject) or ''
