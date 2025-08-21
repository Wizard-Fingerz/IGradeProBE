from django.shortcuts import get_object_or_404
from rest_framework import viewsets, generics, status
from .models import Exam
from .serializers import CreateExamSerializer, ExamSerializer, GetExamDetailSerializer
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import pagination
from rest_framework.authentication import TokenAuthentication
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework import permissions

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

class ExamViewSet(viewsets.ModelViewSet):
    queryset = Exam.objects.all()
    serializer_class = ExamSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination

    @swagger_auto_schema(
        operation_description="Retrieve a list of exams",
        responses={200: openapi.Response('Success', ExamSerializer(many=True))}
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Create a new exam",
        request_body=ExamSerializer,
        responses={201: openapi.Response('Created', ExamSerializer)}
    )
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Retrieve a specific exam",
        responses={200: openapi.Response('Success', ExamSerializer)}
    )
    def retrieve(self, request, *args, **kwargs):
        return super().retrieve(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Update a specific exam",
        request_body=ExamSerializer,
        responses={200: openapi.Response('Updated', ExamSerializer)}
    )
    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)

    @swagger_auto_schema(
        operation_description="Delete a specific exam",
        responses={204: 'No Content'}
    )
    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)



class ExamCreateView(generics.CreateAPIView):
    queryset = Exam.objects.all()
    serializer_class = CreateExamSerializer
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        import copy

        def clean_sub_questions(sub_questions):
            cleaned = []
            for sq in sub_questions:
                # Remove blank or empty fields
                sq_clean = {k: v for k, v in sq.items() if v not in [None, '', [], {}]}
                # Remove sub_questions with all fields empty
                if sq_clean:
                    cleaned.append(sq_clean)
            return cleaned

        def clean_questions(questions):
            cleaned = []
            for q in questions:
                q_clean = {k: v for k, v in q.items() if v not in [None, '', [], {}]}
                # Clean sub_questions if present
                if 'sub_questions' in q_clean and isinstance(q_clean['sub_questions'], list):
                    q_clean['sub_questions'] = clean_sub_questions(q_clean['sub_questions'])
                    # Remove sub_questions if now empty
                    if not q_clean['sub_questions']:
                        q_clean.pop('sub_questions')
                # Remove questions with all fields empty
                if q_clean:
                    cleaned.append(q_clean)
            return cleaned

        data = copy.deepcopy(request.data)
        if 'questions' in data and isinstance(data['questions'], list):
            data['questions'] = clean_questions(data['questions'])

        serializer = self.get_serializer(data=data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Set the examiner to the current user
        serializer.validated_data['created_by'] = request.user

        # Save the exam and associated questions
        self.perform_create(serializer)

        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

class ExamUpdateView(generics.UpdateAPIView):
    queryset = Exam.objects.all()
    serializer_class = CreateExamSerializer
    permission_classes = [permissions.IsAuthenticated]

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        if not serializer.is_valid():
            print(serializer.errors)  # To see detailed errors
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        # Set the examiner to the current user if needed
        serializer.validated_data['created_by'] = request.user

        # Save the exam and associated questions
        self.perform_update(serializer)

        # if getattr(instance, '_prefetched_objects_cache', None):
        #     # If 'prefetch_related' has been applied to a queryset, we need to forcibly
        #     # invalidate the prefetch cache on the instance.
        #     instance._prefetched_objects_cache = {}

        return Response(serializer.data)
    

class ExamDetailView(generics.RetrieveAPIView):
    queryset = Exam.objects.all()
    serializer_class = GetExamDetailSerializer
    permission_classes = [permissions.IsAuthenticated]
