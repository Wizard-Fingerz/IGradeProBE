
from django.db import models

from account.students.models import Student
from app.exams.models import Exam
from app.subjects.models import Subject


class ExamResultScore(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='exam_result_scores')
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    subject = models.ForeignKey(Subject, on_delete=models.CASCADE)
    exam_score = models.IntegerField(null=True, blank=True)
    percentage_score = models.FloatField(null=True, blank=True)
    grade = models.CharField(max_length=2, blank=True)
    is_disabled = models.BooleanField(default=True)
    effective_total_marks = models.IntegerField(null=True, blank=True)  # New field

    @property
    def exam_total_mark(self):
        return self.exam.total_mark if self.exam else 0

    @property
    def student_name(self):
        return self.student.get_full_name()
    
    @property
    def subject_detials(self):
        return f"{self.subject.name} - {self.subject.code}"

    @property
    def student_detials(self):
        return f"{self.student.get_full_name()} - {self.student.candidate_number}"
    
    @property
    def examination_number(self):
        return self.student.examination_number



    def __str__(self):
        return f"{self.student}'s exam score for {self.subject}"

    def calculate_grade(self):
        if self.effective_total_marks == 0:
            self.percentage_score = 0
        else:
            # percent_score = (self.exam_score / self.effective_total_marks) * 100
            percent_score = (self.exam_score / self.exam_total_mark) * 100
            # self.percentage_score = percent_score
            self.percentage_score = round(percent_score)  # Round to the nearest integer       
            if percent_score >= 75:
                self.grade = 'A1'
            elif percent_score >= 70:
                self.grade = 'B2'
            elif percent_score >= 65:
                self.grade = 'B3'
            elif percent_score >= 60:
                self.grade = 'C4'
            elif percent_score >= 55:
                self.grade = 'C5'
            elif percent_score >= 50:
                self.grade = 'C6'
            elif percent_score >= 45:
                self.grade = 'D7'
            elif percent_score >= 40:
                self.grade = 'E8'
            else:
                self.grade = 'F9'

