from rest_framework.views import APIView
from rest_framework.response import Response
from account.students.models import Student
from app.subjects.models import Subject
from app.ocr.models import StudentScript
from app.results.models import ExamResult  # Assuming you have a Result model

class AnalyticsView(APIView):
    def get(self, request, *args, **kwargs):
        total_students = Student.objects.count()
        total_subjects = Subject.objects.count()
        total_exam_graded = StudentScript.objects.count()
        total_results = ExamResult.objects.count()

        data = {
            "total_students": total_students,
            "total_subjects": total_subjects,
            "total_exam_graded": total_exam_graded,
            "total_results": total_results,
        }

        return Response(data)