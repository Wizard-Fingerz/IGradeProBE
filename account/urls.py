from django.urls import path, include
from rest_framework.routers import DefaultRouter
from account.students.views import StudentViewSet
from .views import LogoutView, UserViewSet
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')  # Explicitly set 'users' as the prefix
router.register(r'students', StudentViewSet, basename='students')  # Students endpoint

urlpatterns = [
    path('account/', include(router.urls)),  # Include all router URLs under 'account/'
    path('login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),  # JWT login
    path('login/refresh/', TokenRefreshView.as_view(), name='token_refresh'),  # JWT token refresh
     path('logout/', LogoutView.as_view(), name='logout'),

]
