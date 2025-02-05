from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save
from django.dispatch import receiver
from .managers import UserManager
from rest_framework.authtoken.models import Token

# Create your models here.
class User(AbstractUser):
    is_admin = models.BooleanField(default=False, verbose_name='Admin')
    is_student = models.BooleanField(default=False, verbose_name='Student')
    is_examiner = models.BooleanField(default=False, verbose_name='Examiner')
    student_id = models.CharField(max_length=150, unique=True,blank = True,null = True)
    other_name = models.CharField(max_length=150, blank = True, null = True)
    examiner_id = models.CharField(max_length=150, unique=True, blank = True,null = True)
    
    class Meta:
        swappable = 'AUTH_USER_MODEL'
        
    objects = UserManager()

    def __str__(self):
        return str(self.username) or ''

@receiver(post_save, sender=User)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    """
    Signal handler to create a token for the user when a new user is created.
    """
    if created:
        Token.objects.create(user=instance)