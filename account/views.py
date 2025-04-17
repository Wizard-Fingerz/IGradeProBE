from django.shortcuts import render
from .models import *
import csv
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework import viewsets
from .serializers import *
from rest_framework import generics
from rest_framework.response import Response
from rest_framework import generics, permissions, viewsets, views
from rest_framework.authentication import TokenAuthentication
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.generics import RetrieveUpdateDestroyAPIView
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.contrib.auth.hashers import make_password
from django.http import JsonResponse, HttpResponse


from rest_framework.decorators import action




class CustomObtainAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(
            data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)

        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)

        # Determine the type of user (admin, user, examiner)
        user_type = None

        if user.is_admin:
            user_type = 'admin'
        elif user.is_examiner:
            user_type = 'examiner'

        return Response({
            'token': token.key,
            'user_type': user_type,

        })

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()

    def get_serializer_class(self):
        if self.request.method == 'GET':
            return UserSerializer
        else:
            return UserSerializer
        

    @action(methods=['get'], detail=False, url_path='me', url_name='me')
    def get_current_user(self, request):
        try:
            # Try to fetch the authenticated user
            user = request.user
            # If user is found, serialize and return the data
            serializer = self.get_serializer(user)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except User.DoesNotExist:
            # If no user found (which is unlikely as we're using request.user), return a 404
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)
 
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.authtoken.models import Token

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        try:
            # Delete the user's token to log them out
            request.user.auth_token.delete()
            return Response({"message": "Successfully logged out."}, status=status.HTTP_200_OK)
        except Token.DoesNotExist:
            return Response({"error": "Invalid token or user not logged in."}, status=status.HTTP_400_BAD_REQUEST)