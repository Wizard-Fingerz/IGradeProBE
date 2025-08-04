from django.db.models import Sum
import textdistance
import re
from app.exams.models import Exam
from app.ocr.easy_ocr import handwritten_to_text_easyocr
from app.ocr.my_script_ocr import extract_text_from_image
from app.ocr.prediction import PredictionService
from app.questions.models import SubjectQuestion
from app.results.models import ExamResult
from app.scores.models import ExamResultScore
from app.subjects.models import Subject  # Import the Subject model
from account.students.models import Student
import csv
import zipfile
import os
from django.core.files.base import ContentFile
from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status

from app.test_ocr_main.main import extract_text_with_test_ocr
from .models import StudentScript, ScriptPage
from rest_framework.decorators import action
from .serializers import StudentScriptSerializer
from .google_ocr_modified import detect_document_modified, extract_all_text_between_as_ae, find_matching_question


class StudentScriptViewSet(viewsets.ModelViewSet):
    queryset = StudentScript.objects.all()
    serializer_class = StudentScriptSerializer
    parser_classes = [MultiPartParser]

    def create(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get("file")
        subject_id = request.data.get("subject_id")
        student_id = request.data.get("student_id")

        if not uploaded_file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        if not subject_id:
            return Response({"error": "No subject provided"}, status=status.HTTP_400_BAD_REQUEST)

        if not student_id:
            return Response({"error": "No student provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Get the subject
        try:
            subject = Subject.objects.get(id=subject_id)
        except Subject.DoesNotExist:
            return Response({"error": "Invalid subject ID"}, status=status.HTTP_400_BAD_REQUEST)

        # Get the student
        try:
            student = Student.objects.get(id=student_id)
        except Student.DoesNotExist:
            return Response({"error": "Invalid student ID"}, status=status.HTTP_400_BAD_REQUEST)

        # Save ZIP file temporarily
        zip_path = default_storage.save(
            f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        zip_full_path = default_storage.path(zip_path)

        try:
            with zipfile.ZipFile(zip_full_path, "r") as zip_ref:
                # Extract and read the CSV file
                csv_file_name = [
                    f for f in zip_ref.namelist() if f.endswith(".csv")][0]
                csv_file_data = zip_ref.read(
                    csv_file_name).decode("utf-8").splitlines()
                csv_reader = csv.DictReader(csv_file_data)

                # Extract images and associate them with student data
                for row in csv_reader:
                    # Get or create the Student object
                    student, created = Student.objects.get_or_create(
                        center_number=row["centre_number"],
                        candidate_number=row["candidate_number"],
                        examination_number=row["examination_number"],
                        defaults={
                            "exam_type": row.get("exam_type", None),
                            "year": row.get("year", None),
                        },
                    )

                    # Create a StudentScript instance for this student and subject
                    student_script = StudentScript.objects.create(
                        student_id=student,
                        subject=subject,
                    )

                    # Process images for this student
                    for file_name in zip_ref.namelist():
                        if file_name.endswith((".png", ".jpg", ".jpeg")) and f"{row['examination_number']}" in file_name:
                            file_data = zip_ref.read(file_name)
                            image_path = f"scripts/{row['examination_number']}/{file_name}"
                            default_storage.save(
                                image_path, ContentFile(file_data))

                            # Perform OCR
                            extracted_text = detect_document_modified(
                                default_storage.path(image_path), settings.GOOGLE_APPLICATION_CREDENTIALS)

                            # Create a ScriptPage instance
                            ScriptPage.objects.create(
                                script=student_script,
                                image=image_path,
                                extracted_text=extracted_text,
                                # Ensure page_number is set
                                page_number=row.get('page_number', 1)
                            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            os.remove(zip_full_path)  # Cleanup temporary ZIP file

        return Response({"message": "Upload and processing complete"}, status=status.HTTP_201_CREATED)

    @action(detail=False, methods=['get'], url_path='by-student/(?P<student_id>[^/.]+)')
    def get_scripts_by_student_id(self, request, student_id=None):
        scripts = StudentScript.objects.filter(student_id=student_id)
        if not scripts.exists():
            return Response({"error": "No scripts found for the given student ID"}, status=status.HTTP_404_NOT_FOUND)

        serializer = self.get_serializer(scripts, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class UploadScriptView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        student_id = request.data.get("student_id")
        uploaded_file = request.FILES.get("file")

        if not uploaded_file or not student_id:
            return Response({"error": "Missing file or student_id"}, status=status.HTTP_400_BAD_REQUEST)

        # Create StudentScript instance
        script_instance = StudentScript.objects.create(student_id=student_id)

        # Save ZIP file temporarily
        zip_path = default_storage.save(
            f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        zip_full_path = default_storage.path(zip_path)

        # Extract images
        extracted_pages = []
        try:
            with zipfile.ZipFile(zip_full_path, "r") as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        file_data = zip_ref.read(file_name)
                        image_path = f"scripts/{student_id}/{file_name}"
                        default_storage.save(
                            image_path, ContentFile(file_data))

                        # Perform OCR
                        extracted_text = detect_document_modified(
                            default_storage.path(image_path))

                        # Save in DB
                        page_instance = ScriptPage.objects.create(
                            script=script_instance,
                            image=image_path,
                            extracted_text=extracted_text
                        )
                        extracted_pages.append(page_instance)
        finally:
            os.remove(zip_full_path)  # Cleanup

        return Response(StudentScriptSerializer(script_instance).data, status=status.HTTP_201_CREATED)


class BulkUploadScriptView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        uploaded_file = request.FILES.get("file")
        subject_id = request.data.get("subject_id")

        if not uploaded_file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        if not subject_id:
            return Response({"error": "No subject provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Get the subject
        try:
            subject = Subject.objects.get(id=subject_id)
        except Subject.DoesNotExist:
            return Response({"error": "Invalid subject ID"}, status=status.HTTP_400_BAD_REQUEST)

        # Save ZIP file temporarily
        zip_path = default_storage.save(
            f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        zip_full_path = default_storage.path(zip_path)

        # try:
        with zipfile.ZipFile(zip_full_path, "r") as zip_ref:
            # Extract and read the CSV file
            csv_file_name = [
                f for f in zip_ref.namelist() if f.endswith(".csv")][0]
            csv_file_data = zip_ref.read(
                csv_file_name).decode("utf-8").splitlines()
            csv_reader = csv.DictReader(csv_file_data)

            for row in csv_reader:
                student, created = Student.objects.get_or_create(
                    center_number=row["centre_number"],
                    candidate_number=row["candidate_number"],
                    examination_number=row["examination_number"],
                    defaults={
                        "exam_type": row.get("exam_type", None),
                        "year": row.get("year", None),
                    },
                )

                student_script = StudentScript.objects.create(
                    student_id=student,
                    subject=subject,
                )

                for file_name in zip_ref.namelist():
                    if file_name.endswith((".png", ".jpg", ".jpeg")) and f"{row['examination_number']}" in file_name:
                        file_data = zip_ref.read(file_name)
                        image_path = f"scripts/{row['examination_number']}/{file_name}"
                        default_storage.save(
                            image_path, ContentFile(file_data))
                        
                        print(f"Processing image: {image_path}")

                        # extracted_text = detect_document_modified(
                        #     default_storage.path(
                        #         image_path), settings.GOOGLE_APPLICATION_CREDENTIALS
                        # )

                        extracted_text = extract_text_with_test_ocr(default_storage.path(image_path))

                        print(f"Extracted text: {extracted_text}")

                        extracted_text = extract_all_text_between_as_ae(
                            extracted_text)
                        # extracted_text = extract_text_from_image(default_storage.path(image_path))
                        # print(f"Extracted text from MY Script: {extracted_text}")

                        # if extracted_text == None:
                        #     extracted_text = extract_text_from_image(image_path)
                        #     print("No text extracted from image.")
                        #     print(f"Extracted text from MY Script: {extracted_text}")


                        # extracted_text = handwritten_to_text_easyocr(
                        #     default_storage.path(image_path))
                        # Iterate over extracted Q&A
                        for qa in extracted_text:
                            question_text = qa.get("question")
                            student_answer = qa.get("answer")
                            # print(qa)
                            # print(f"Question: {question_text}")
                            # print(f"First Answer Extracted: {student_answer}")

                            if question_text and student_answer:
                                question = find_matching_question(
                                    question_text)
                                # print(f"Question: {question_text}")
                                # print(f"Answer: {student_answer}")
                                print(f"Question: {question}")

                                if question:
                                    self.grade_answer(
                                        student, question, student_answer)

            
        # except Exception as e:
        #     return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        # finally:
        #     os.remove(zip_full_path)  # Cleanup temporary ZIP file

        return Response({"message": "Upload and processing complete"}, status=status.HTTP_201_CREATED)

    def grade_answer(self, student, question, student_answer):
        prediction_service = PredictionService()
        student_score = prediction_service.predict(
            question_id=question.id,
            comprehension=question.comprehension,
            question=question.question,
            question_score=question.question_score,
            examiner_answer=question.examiner_answer,
            student_answer=student_answer
        )

        print(
            f"Grading student {student.id} for question '{question.question}'")
        print(f"Student answer: {student_answer}")
        print(f"Score awarded: {student_score}")

        exam = Exam.objects.get(subject = question.subject)

        exam_result, created = ExamResult.objects.get_or_create(
            exam = exam,
            student=student,
            question=question,
            defaults={
                'student_answer': student_answer,
                'student_score': student_score,
                'attempted': True
            }
        )

        if not created:
            exam_result.student_answer = student_answer
            exam_result.student_score = student_score
            exam_result.attempted = True
            exam_result.save()

        self.detect_plagiarism(question.id, student_answer, exam_result)

        # Ensure question.subject is a related instance
        if isinstance(question.subject, str):
            subject_instance = SubjectQuestion.objects.filter(
                name=question.subject).first()
        else:
            subject_instance = question.subject

        if subject_instance:
            total_score = ExamResult.objects.filter(student=student, question__subject=subject_instance).aggregate(
                Sum('student_score')
            )['student_score__sum'] or 0
        else:
            total_score = 0  # Avoid query failure if no subject is found

        print(f"Total score for student {student.id}: {total_score}")

        exam_result_score, _ = ExamResultScore.objects.get_or_create(
            student=student, subject=question.subject,

        )
        exam_result_score.exam_score = total_score
        exam_result_score.calculate_grade()
        exam_result_score.save()

        print(f"Total score for student {student.id}: {total_score}")

    def detect_plagiarism(self, question_id, new_answer, new_exam_result):
        existing_results = ExamResult.objects.filter(
            question_id=question_id).exclude(id=new_exam_result.id)

        for result in existing_results:
            similarity = textdistance.jaccard(
                new_answer, result.student_answer)

            new_exam_result.similarity_score = similarity
            result.similarity_score = similarity
            new_exam_result.save()
            result.save()


class ScriptOutputView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            # Fetch all scripts
            scripts = StudentScript.objects.prefetch_related(
                "pages", "subject").all().order_by('-uploaded_at')
            serializer = StudentScriptSerializer(scripts, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
