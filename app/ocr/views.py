from account.students.models import Student
import csv
import zipfile
import os
from django.core.files.base import ContentFile
from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from .models import StudentScript, ScriptPage
from .serializers import StudentScriptSerializer
from .google_ocr_modified import detect_document_modified


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
        if not uploaded_file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

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

                    # Create a StudentScript instance for this student
                    student_script = StudentScript.objects.create(
                        student_id=student)

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
                            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            os.remove(zip_full_path)  # Cleanup temporary ZIP file

        return Response({"message": "Upload and processing complete"}, status=status.HTTP_201_CREATED)


class ScriptOutputView(APIView):
    def get(self, request, *args, **kwargs):
        try:
            # Fetch all scripts
            scripts = StudentScript.objects.prefetch_related("pages").all()
            serializer = StudentScriptSerializer(scripts, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)