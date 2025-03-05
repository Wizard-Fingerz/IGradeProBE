from rest_framework import viewsets
from .models import ExamResultScore
from .serializers import ExamResultScoreSerializer
from django.shortcuts import get_object_or_404
from rest_framework import viewsets, generics, status
from rest_framework.response import Response
from rest_framework import pagination


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




class ExamResultScoreViewSet(viewsets.ModelViewSet):
    serializer_class = ExamResultScoreSerializer
    pagination_class = CustomPagination


    def get_queryset(self):
        # Filter the queryset to retrieve only the exam result scores of the currently logged-in student
        return ExamResultScore.objects.all()
