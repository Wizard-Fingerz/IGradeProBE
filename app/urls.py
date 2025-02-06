from django.urls import path, include
from rest_framework.routers import DefaultRouter
from app.exams.views import ExamCreateView, ExamDetailView, ExamUpdateView, ExamViewSet
from app.ocr.views import BulkUploadScriptView, ScriptOutputView, UploadScriptView
from app.questions.views import SubjectQuestionViewSet
from app.subjects.views import SubjectViewSet


app_name = 'app'

router = DefaultRouter()
router.register(r'exams', ExamViewSet, basename='exams')
router.register(r'subjects', SubjectViewSet, basename='subjects')
# router.register(r'exam-create', ExamCreateView, basename='create-exam')
router.register(r'subject-questions', SubjectQuestionViewSet,
                basename='subject-question')


urlpatterns = [

    path('app/', include(router.urls)),
    path('app/exam-create/', ExamCreateView.as_view(), name='create-exam'),
    path('app/exam-update/<int:pk>/',
         ExamUpdateView.as_view(), name='update-exam'),

    path('app/exam-details/<int:pk>/',
         ExamDetailView.as_view(), name='exam-detail'),

    path('upload-script/', UploadScriptView.as_view(), name='upload-script'),
    path('app/bulk-upload-script/', BulkUploadScriptView.as_view(),
         name='bulk-upload-script'),
    path("app/scripts", ScriptOutputView.as_view(), name="script-output"),


]

# Add the router URLs at the end
urlpatterns += router.urls
