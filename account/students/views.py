import os
from rest_framework import viewsets
from account.students.tasks import process_student_upload
from .models import Student
from .serializers import DetailedStudentSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import pagination
from rest_framework.authentication import TokenAuthentication
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import csv
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.decorators import action
from rest_framework import status
from .models import Student
from .serializers import DetailedStudentSerializer
from django.core.exceptions import ValidationError
from django.core.files.storage import FileSystemStorage
from django.conf import settings


class CustomPagination(pagination.PageNumberPagination):
    page_size = 15
    page_size_query_param = 'page_size'
    max_page_size = 15

    def get_paginated_response(self, data):
        return Response({
            'count': self.page.paginator.count,
            'next': self.get_next_link(),
            'previous': self.get_previous_link(),
            'num_pages': self.page.paginator.num_pages,
            'page_size': self.page_size,
            'current_page': self.page.number,
            'results': data
        })

class StudentViewSet(viewsets.ModelViewSet):
    queryset = Student.objects.all()
    serializer_class = DetailedStudentSerializer
    pagination_class = CustomPagination

    @swagger_auto_schema(
        operation_description="Retrieve a list of Students",
        responses={200: openapi.Response('Success', DetailedStudentSerializer(many=True))}
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Create a new Student",
        request_body=DetailedStudentSerializer,
        responses={201: openapi.Response('Created', DetailedStudentSerializer)}
    )
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Retrieve a specific Student",
        responses={200: openapi.Response('Success', DetailedStudentSerializer)}
    )
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Update a specific Student",
        request_body=DetailedStudentSerializer,
        responses={200: openapi.Response('Updated', DetailedStudentSerializer)}
    )
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Delete a specific Student",
        responses={204: 'No Content'}
    )
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
    
    
    @action(detail=False, methods=['post'], url_path='bulk-upload')
    @swagger_auto_schema(
        operation_description="Bulk upload Students via CSV file",
        request_body=openapi.Schema(type=openapi.TYPE_OBJECT, properties={
            'file': openapi.Schema(type=openapi.TYPE_FILE)
        }),
        responses={
            200: openapi.Response('CSV Processed Successfully'),
            400: openapi.Response('Bad Request: Invalid File or Format'),
        }
    )
    # @action(detail=False, methods=['post'], url_path='bulk-upload')
    def bulk_upload(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"detail": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        
        # Ensure the file is a CSV
        if not file.name.endswith('.csv'):
            return Response({"detail": "Invalid file format. Please upload a CSV file."},
                            status=status.HTTP_400_BAD_REQUEST)

        fs = FileSystemStorage()
        filename = fs.save(file.name, file)  # Save the file
        file_path = fs.path(filename)  # Get the absolute file path correctly

        # Pass the absolute path to the Celery task
        process_student_upload.delay(filename)

        return Response({"detail": "CSV processed and students upload has been initiated."}, status=status.HTTP_200_OK)

        # except Exception as e:
        #     return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
