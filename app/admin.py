from django.contrib import admin

from import_export.admin import ImportExportModelAdmin

from app.subjects.models import Subject

# Register your models here.


# Register your models here.
@admin.register(Subject)
class UsersAdmin(ImportExportModelAdmin):
    list_display = ('name','code','description')
    search_fields = ['name',]
    list_filter = ['name',]


admin.site.site_header = 'Intelligent Essay Grading Pro Administration Dashboard'
