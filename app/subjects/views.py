from rest_framework import viewsets
from .models import Subject
from .serializers import SubjectSerializer
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
from .models import Subject
from .serializers import SubjectSerializer
from django.core.exceptions import ValidationError

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

class SubjectViewSet(viewsets.ModelViewSet):
    queryset = Subject.objects.all()
    serializer_class = SubjectSerializer
    pagination_class = CustomPagination

    @swagger_auto_schema(
        operation_description="Retrieve a list of subjects",
        responses={200: openapi.Response('Success', SubjectSerializer(many=True))}
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Create a new subject",
        request_body=SubjectSerializer,
        responses={201: openapi.Response('Created', SubjectSerializer)}
    )
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Retrieve a specific subject",
        responses={200: openapi.Response('Success', SubjectSerializer)}
    )
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Update a specific subject",
        request_body=SubjectSerializer,
        responses={200: openapi.Response('Updated', SubjectSerializer)}
    )
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Delete a specific subject",
        responses={204: 'No Content'}
    )
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)
    
    
    @action(detail=False, methods=['post'], url_path='bulk-upload')
    @swagger_auto_schema(
        operation_description="Bulk upload subjects via CSV file",
        request_body=openapi.Schema(type=openapi.TYPE_OBJECT, properties={
            'file': openapi.Schema(type=openapi.TYPE_FILE)
        }),
        responses={
            200: openapi.Response('CSV Processed Successfully'),
            400: openapi.Response('Bad Request: Invalid File or Format'),
        }
    )
    def bulk_upload(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({"detail": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        
        # Ensure the file is a CSV
        if not file.name.endswith('.csv'):
            return Response({"detail": "Invalid file format. Please upload a CSV file."},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            # Read CSV file
            csv_file = file.read().decode('utf-8').splitlines()
            reader = csv.DictReader(csv_file)

            subjects_to_create = []
            for row in reader:
                # Extract necessary data from the CSV row and validate
                subject_data = {
                    'name': row.get('name'),
                    'code': row.get('code'),
                    'description': row.get('description'),
                }

                # Validate required fields
                if not subject_data['name'] or not subject_data['code']:
                    raise ValidationError(f"Missing required field in CSV row: {row}")

                # Create subject instance to add to bulk create list
                subjects_to_create.append(Subject(**subject_data))

            # Bulk create subjects
            Subject.objects.bulk_create(subjects_to_create)
            return Response({"detail": "CSV processed and subjects uploaded successfully."}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": str(e)}, status=status.HTTP_400_BAD_REQUEST)
