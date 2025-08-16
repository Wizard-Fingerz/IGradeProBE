from django.db import models
from account.students.models import Student
from app.exams.models import Exam
from app.questions.models import SubjectQuestion
from app.scores.models import ExamResultScore
from django.db.models import Sum


class ExamResult(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='exam_results')
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    question = models.ForeignKey(SubjectQuestion, on_delete=models.CASCADE)
    student_answer = models.TextField(null=True, blank=True)
    student_score = models.IntegerField(null=True, blank=True)
    similarity_score = models.FloatField(null=True, blank=True)  # New field to store similarity score
    attempted = models.BooleanField(default=True)  # New field to indicate if the question was attempted
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def question_text(self):
        return self.question.question


    @property
    def candidate_number(self):
        return self.student.candidate_number
    
    @property
    def examination_number(self):
        return self.student.examination_number

    @property
    def question_number(self):
        return self.question.question_number
    
    
    @property
    def question_score(self):
        return self.question.question_score

    @property
    def examiner_answer(self):
        return self.question.examiner_answer

    
    @property
    def exam_comprehension(self):
        return self.question.comprehension
    
    def __str__(self):
        return f"{self.student.candidate_number}'s answer to {self.question}"
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.update_exam_result_score()

    def update_exam_result_score(self):
        # Calculate the total score for attempted compulsory questions and attempted optional questions
        results = ExamResult.objects.filter(student=self.student, question__subject=self.question.subject)
        compulsory_questions = results.filter(question__is_optional=False)
        optional_questions = results.filter(question__is_optional=True, attempted=True)

        # Calculate total scores
        compulsory_total_score = compulsory_questions.aggregate(total_score=Sum('student_score'))['total_score'] or 0
        optional_total_score = optional_questions.aggregate(total_score=Sum('student_score'))['total_score'] or 0

        total_score = compulsory_total_score + optional_total_score

        # Calculate the effective total mark
        compulsory_total_marks = compulsory_questions.aggregate(total_marks=Sum('question__question_score'))['total_marks'] or 0
        optional_total_marks = optional_questions.aggregate(total_marks=Sum('question__question_score'))['total_marks'] or 0
        effective_total_marks = compulsory_total_marks + optional_total_marks

        # Update or create ExamResultScore instance for the student
        exam_result_score, _ = ExamResultScore.objects.get_or_create(student=self.student, subject=self.question.subject, exam = self.exam)
        exam_result_score.exam_score = total_score
        exam_result_score.effective_total_marks = effective_total_marks
        exam_result_score.calculate_grade()
        exam_result_score.save()
