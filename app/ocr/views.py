from django.db.models import Sum
import textdistance
import re
from app.exams.models import Exam
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
from .google_ocr_modified import detect_document_modified, extract_all_text_between_as_ae, extract_all_text_sequentially, find_matching_question


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

        try:
            subject = Subject.objects.get(id=subject_id)
        except Subject.DoesNotExist:
            return Response({"error": "Invalid subject ID"}, status=status.HTTP_400_BAD_REQUEST)

        zip_path = default_storage.save(
            f"temp/{uploaded_file.name}", ContentFile(uploaded_file.read()))
        zip_full_path = default_storage.path(zip_path)

        try:
            
            
            with zipfile.ZipFile(zip_full_path, "r") as zip_ref:
                # Load CSV into memory
                csv_file_name = [f for f in zip_ref.namelist() if f.endswith(".csv")][0]
                csv_file_data = zip_ref.read(csv_file_name).decode("utf-8").splitlines()
                csv_rows = list(csv.DictReader(csv_file_data))

                # Get list of all exam numbers
                exam_numbers = {row["examination_number"] for row in csv_rows}

                # Step 1: Build mapping of exam_number -> unique image file names
                image_map = {exam_num: set() for exam_num in exam_numbers}
                for file_name in zip_ref.namelist():
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        for exam_num in exam_numbers:
                            if f"{exam_num}" in file_name:
                                image_map[exam_num].add(file_name)  # set avoids duplicates

                # Step 2: Process each student once
                # Step 2: Process each student once
                processed_students = set()

                grading_count = 0  # Add this before starting the main for loop

                for row in csv_rows:
                    exam_num = row["examination_number"]

                    # Skip if already processed
                    if exam_num in processed_students:
                        continue

                    processed_students.add(exam_num)

                    student, _ = Student.objects.get_or_create(
                        center_number=row["centre_number"],
                        candidate_number=row["candidate_number"],
                        examination_number=exam_num,
                        defaults={
                            "exam_type": row.get("exam_type"),
                            "year": row.get("year"),
                        },
                    )

                    try:
                        exam = Exam.objects.get(subject=subject)
                    except Exam.DoesNotExist:
                        continue

                    StudentScript.objects.create(student_id=student, subject=subject)

                    # Combine OCR output for all images for this student
                    combined_text = ""
                    for file_name in sorted(image_map.get(exam_num, [])):
                        file_data = zip_ref.read(file_name)
                        image_path = f"scripts/{exam_num}/{os.path.basename(file_name)}"

                        if not default_storage.exists(image_path):
                            default_storage.save(image_path, ContentFile(file_data))

                        # extracted_text = process_ocr_pipeline(default_storage.path(image_path))
                                                    # Perform OCR
                        extracted_text = detect_document_modified(
                            default_storage.path(image_path), settings.GOOGLE_APPLICATION_CREDENTIALS)

                        combined_text += "\n" + extracted_text

                    # Extract and grade answers
                    extracted_text_from_regex = extract_all_text_sequentially(combined_text)

                    for qa in extracted_text_from_regex:
                        question_text = qa.get("question")
                        student_answer = qa.get("answer")
                        if question_text and student_answer:
                            question = find_matching_question(question_text)
                            if question:
                                self.grade_answer(student, question, student_answer)
                                grading_count += 1  # Increment counter
               
                print(f"✅ Total grading operations performed: {grading_count}")



        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            os.remove(zip_full_path)

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

        # print(
        #     f"Grading student {student.id} for question '{question.question}'")
        # print(f"Student answer: {student_answer}")
        # print(f"Score awarded: {student_score}")

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

        # self.detect_plagiarism(question.id, student_answer, exam_result)

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

        # print(f"Total score for student {student.id}: {total_score}")

        exam_result_score, _ = ExamResultScore.objects.get_or_create(
            student=student, subject=question.subject,

        )
        exam_result_score.exam_score = total_score
        exam_result_score.calculate_grade()
        exam_result_score.save()

        print(f"Total score for student {student.id}: {total_score}")

    
    def update_parent_question_scores(self, student, exam):
        """
        Aggregate scores from sub-questions and assign to parent questions.
        """
        parent_questions = SubjectQuestion.objects.filter(sub_questions__isnull=False).distinct()

        for parent in parent_questions:
            sub_results = ExamResult.objects.filter(
                student=student,
                exam=exam,
                question__parent_question=parent
            )

            total_score = sub_results.aggregate(Sum("student_score"))["student_score__sum"] or 0

            parent_result, created = ExamResult.objects.get_or_create(
                student=student,
                exam=exam,
                question=parent,
                defaults={"student_score": total_score, "attempted": True}
            )

            if not created:
                parent_result.student_score = total_score
                parent_result.attempted = True
                parent_result.save()


    
    # def detect_plagiarism(self, question_id, new_answer, new_exam_result):
    #     existing_results = ExamResult.objects.filter(
    #         question_id=question_id).exclude(id=new_exam_result.id)

    #     for result in existing_results:
    #         similarity = textdistance.jaccard(
    #             new_answer, result.student_answer)

    #         new_exam_result.similarity_score = similarity
    #         result.similarity_score = similarity
    #         new_exam_result.save()
    #         result.save()



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
