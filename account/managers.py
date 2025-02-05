from django.contrib.auth.models import BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, username, password=None, **extra_fields):
        user = self.model(username=username, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_student(self, username, password=None, **extra_fields):
        extra_fields.setdefault('is_student', True)
        return self.create_user(username, password, **extra_fields)
    
    def create_examiner(self, username, password=None, **extra_fields):
        extra_fields.setdefault('is_examiner', True)
        return self.create_user(username, password, **extra_fields)
    
    def create_superuser(self, username, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        return self.create_user(username, password, **extra_fields)
