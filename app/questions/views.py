from rest_framework import viewsets
from .models import SubjectQuestion
from .serializers import SubjectQuestionSerializer, SubjectQuestionNestedSerializer
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

class SubjectQuestionViewSet(viewsets.ModelViewSet):
    queryset = SubjectQuestion.objects.all()
    serializer_class = SubjectQuestionSerializer
    pagination_class = CustomPagination

    def get_serializer_class(self):
        if self.action in ['create', 'update', 'partial_update']:
            return SubjectQuestionNestedSerializer
        return SubjectQuestionSerializer