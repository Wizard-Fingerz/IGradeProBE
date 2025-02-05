from django.db import models

class Notification(models.Model):
    ERROR = 'error'
    SUCCESS = 'success'
    WARNING = 'warning'
    
    NOTIFICATION_CHOICES = [
        (ERROR, 'Error'),
        (SUCCESS, 'Success'),
        (WARNING, 'Warning'),
    ]
    
    level = models.CharField(max_length=10, choices=NOTIFICATION_CHOICES, default=ERROR)
    message = models.TextField()
    task_name = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    extra_data = models.JSONField(null=True, blank=True)  # Store additional data, like file_path, etc.
    
    def __str__(self):
        return f"{self.level.upper()}: {self.task_name} - {self.message[:30]}"
