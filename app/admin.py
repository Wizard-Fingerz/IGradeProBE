from django.contrib import admin

from import_export.admin import ImportExportModelAdmin

from app.results.models import ExamResult
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


admin.site.site_header = 'Intelligent Essay Grading Pro Administration Dashboard'
