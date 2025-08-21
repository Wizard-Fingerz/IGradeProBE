from django.db import models
from django.db.models import Sum


from app.subjects.models import Subject


class SubjectQuestion(models.Model):
    # exam = models.ForeignKey('Exam', on_delete=models.CASCADE, related_name='exam_questions', null = True, blank = True)
    subject = models.ForeignKey(
        Subject, on_delete=models.CASCADE, null=True, blank=True)
    question_number = models.CharField(max_length=250, blank=True, null=True)
    comprehension = models.TextField(blank = True, null = True)
    question = models.CharField(max_length=1000, blank = True, null = True)
    examiner_answer = models.TextField(blank = True, null = True)
    question_score = models.IntegerField(blank = True, null = True)
    is_optional = models.BooleanField(default=False)
    parent_question = models.ForeignKey(
        'self', on_delete=models.CASCADE, null=True, blank=True, related_name='sub_questions')

    @property
    def subject_name(self):
        return self.subject.name
    
    @property
    def subject_code(self):
        return self.subject.code

    def __str__(self):
        return str(self.question) or ''
