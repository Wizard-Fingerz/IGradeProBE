import uuid
import string
import random
from django.db import models
from account.models import User
from django.contrib.auth.hashers import make_password

class Student(User):
    center_number = models.CharField(max_length=150, blank=True, null=True)
    candidate_number = models.CharField(max_length=150, blank=True, null=True)
    examination_number = models.CharField(max_length=150, unique=True, blank=True, null=True)
    exam_type = models.CharField(max_length=150, blank=True, null=True)
    year = models.BigIntegerField(blank=True, null=True)

    def __str__(self):
        return str(self.username) or ""

    def save(self, *args, **kwargs):
        # Generate a unique username using UUID
        if not self.username:
            self.username = f"student_{uuid.uuid4().hex[:12]}"  # 12-character unique identifier
        
        # Generate a random password if not provided
        if not self.password:
            raw_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            self.password = make_password(raw_password)  # Hash the password
        
        super().save(*args, **kwargs)
