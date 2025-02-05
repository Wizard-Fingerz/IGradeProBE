import time
import csv
import os
from app.notifications.models import Notification
from celery import shared_task
from django.core.files.storage import default_storage
from django.core.exceptions import ValidationError
from django.conf import settings
from .models import Student

@shared_task(bind=True, autoretry_for=(FileNotFoundError,), retry_backoff=3, max_retries=5)
def process_student_upload(self, file_name):

    try:
        """Celery task to process student uploads using Django file storage."""

        # Get the absolute path of the file in MEDIA_ROOT
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        # Ensure file is available before processing (max wait: 10s)
        retries = 0
        while retries < 10:
            if os.path.exists(file_path):
                break
            time.sleep(1)  # Wait 1 second before retrying
            retries += 1

        # Final check before processing
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found in Django storage: {file_path}")

        # Open the file directly using the absolute path
        # Open CSV file and process
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0

            for row in reader:
                required_fields = ['first_name', 'last_name', 'center_number', 'candidate_number', 'examination_number', 'exam_type', 'year']
                
                # Ensure all required fields are present
                if any(row.get(field) is None or row.get(field) == '' for field in required_fields):
                    raise ValidationError(f"Missing required field in CSV row: {row}")

                student = Student(
                    first_name=row.get('first_name'),
                    other_name=row.get('other_name', ''),  # Optional field
                    last_name=row.get('last_name'),
                    center_number=row.get('center_number'),
                    candidate_number=row.get('candidate_number'),
                    examination_number=row.get('examination_number'),
                    exam_type=row.get('exam_type'),
                    year=row.get('year'),
                )

                student.save()  # ✅ Save each student individually
                count += 1

                handle_success(f"Processed {count} students successfully.", file_path)

        return f"Processed {count} students successfully."

    except Exception as e:
            # Log the error and create a notification record
            handle_error(e, file_path)
            raise self.retry(exc=e) 
    

def handle_error(error, file_path):
    """
    Log the error and store it in the Notification table.
    """
    # Create a notification record for the error
    Notification.objects.create(
        level=Notification.ERROR,
        message=str(error),
        task_name='process_student_upload',
        extra_data={'file_path': file_path}
    )

def handle_success(text, file_path):
    """
    Log the error and store it in the Notification table.
    """
    # Create a notification record for the error
    Notification.objects.create(
        level=Notification.SUCCESS,
        message=str(text),
        task_name='process_student_upload',
        extra_data={'file_path': file_path}
    )