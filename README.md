# IGradeProBE

IGradeProBE is a backend service for managing exams, subjects, and questions. It is built using Django and Django REST Framework.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/IGradeProBE.git
    cd IGradeProBE
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Apply the migrations:

    ```bash
    python manage.py migrate
    ```

5. Create a superuser:

    ```bash
    python manage.py createsuperuser
    ```

6. Run the development server:

    ```bash
    python manage.py runserver
    ```

## Usage

To use the API, you can use tools like [Postman](https://www.postman.com/) or [curl](https://curl.se/). Below are some example endpoints.

## API Endpoints

### Authentication

- `POST /api/auth/login/` - Login a user
- `POST /api/auth/logout/` - Logout a user
- `POST /api/auth/register/` - Register a new user

### Exams

- `GET /app/exams/` - List all exams
- `POST /app/exam-create/` - Create a new exam
- `GET /app/exams/<int:pk>/` - Retrieve a specific exam
- `PUT /app/exam-update/<int:pk>/` - Update a specific exam
- `DELETE /app/exams/<int:pk>/` - Delete a specific exam

### Subjects

- `GET /app/subjects/` - List all subjects
- `POST /app/subjects/` - Create a new subject
- `GET /app/subjects/<int:pk>/` - Retrieve a specific subject
- `PUT /app/subjects/<int:pk>/` - Update a specific subject
- `DELETE /app/subjects/<int:pk>/` - Delete a specific subject

### Questions

- `GET /app/subject-questions/` - List all questions
- `POST /app/subject-questions/` - Create a new question
- `GET /app/subject-questions/<int:pk>/` - Retrieve a specific question
- `PUT /app/subject-questions/<int:pk>/` - Update a specific question
- `DELETE /app/subject-questions/<int:pk>/` - Delete a specific question

## Contributing

Contributions are welcome! Please read the contributing guidelines first.

## License

This project is licensed under the MIT License. See the LICENSE file for details.