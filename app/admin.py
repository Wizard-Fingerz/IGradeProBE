from django.contrib import admin

from import_export.admin import ImportExportModelAdmin

from app.exams.models import Exam
from app.ocr.models import ScriptPage, StudentScript
from app.questions.models import SubjectQuestion
from app.results.models import ExamResult
from app.scores.models import ExamResultScore
from app.subjects.models import Subject

# Register your models here.


# Register your models here.
@admin.register(Subject)
class UsersAdmin(ImportExportModelAdmin):
    list_display = ('name','code','description')
    search_fields = ['name',]
    list_filter = ['name',]

@admin.register(ExamResult)
class ExamResultAdmin(ImportExportModelAdmin):
    list_display = ('student','question','student_answer', 'student_score', 'similarity_score', 'attempted')
    search_fields = ['student',]
    list_filter = ['student',]

@admin.register(Exam)
class ExamAdmin(ImportExportModelAdmin):
    list_display = ('question_count','duration','instruction', 'subject', 'total_mark', 'exam_type', 'exam_year', 'question_count', 'subject_name', 'created_by_name',
                    )
    search_fields = ['subject',]
    list_filter = ['subject',]

@admin.register(SubjectQuestion)
class SubjectQuestionAdmin(ImportExportModelAdmin):
    list_display = ('subject', 'question_number', 'comprehension', 'question', 'examiner_answer', 'question_score', 'is_optional', 'parent_question')
    search_fields = ['subject', 'question_number']
    list_filter = ['subject', 'question_number']


@admin.register(ExamResultScore)
class ExamResultScoreAdmin(ImportExportModelAdmin):
    list_display = ('exam', 'student', 'subject', 'student', 'exam_score', 'percentage_score', 'grade', 'is_disabled', 'effective_total_marks', 'exam_total_mark', 'student_name', 'subject_detials', 'student_detials')
    # search_fields = ['subject', 'question_number']
    # list_filter = ['subject', 'question_number']


# You are getting a circular import error because your StudentScript model uses a ForeignKey to Student,
# which is imported from account.students.models. If account.students.models (or its admin.py) also imports
# anything from app.ocr.models (directly or indirectly), this creates a circular import at Django startup.
# 
# To avoid this, you can register StudentScript using admin.site.register instead of the decorator,
# and define the admin class separately. This delays the model lookup until after all models are loaded.

class StudentScriptAdmin(ImportExportModelAdmin):
    list_display = ('student_id', 'subject', 'uploaded_at')

admin.site.register(StudentScript, StudentScriptAdmin)

@admin.register(ScriptPage)
class ScriptPageScoreAdmin(ImportExportModelAdmin):
    list_display = ('script', 'image', 'extracted_answers', 'page_number')



admin.site.site_header = 'Intelligent Essay Grading Pro Administration Dashboard'
