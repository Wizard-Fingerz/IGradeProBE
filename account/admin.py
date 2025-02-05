from django.contrib import admin

from import_export.admin import ImportExportModelAdmin
from account.models import User

# Register your models here.


# Register your models here.
@admin.register(User)
class UsersAdmin(ImportExportModelAdmin):
    list_display = ('username','first_name', 'last_name', 'is_student', 'is_examiner')
    search_fields = ['username', 'first_name', 'last_name']
    list_filter = ['is_student', 'is_examiner', 'is_admin']


admin.site.site_header = 'Intelligent Essay Grading Pro Administration Dashboard'
