from .prediction import PredictionService

student_answer = """
Sand
Loamy
Clay

# """

# student_answer = """
# Free movement of people and goods – ECOWAS allows citizens to travel and trade easily.
# Collective security – member states unite to fight terrorism and civil wars.
# Boost in regional economy through common market policies.
# Cultural exchange programmes among member nations.
# Support for democratic governance and human rights in the region.
# """


examiner_answer = """

Sand 
Gravel
Silt


"""

comprehension = """

Clay
Sand 
Gravel
Silt
stones
Mineral salts
calcium carbonate 
magnesium carbonate
Oxides of iron
aluminium
Silica
Gravel
Silt


"""


# question = "In five key points, fully developed, highlight the extent to which foreign policy plays a critical role in shaping national development outcomes."
# question = "List five extent to which foreign policy plays a critical role in shaping national development outcomes."
# question = "List 5 Artificial Inteligence"
question = "Name three constituents of the part labelled III."
  
# evaluator = PredictionService()
# score, percent = evaluator.predict(comprehension, question, examiner_answer, student_answer, question_score = 15)

prediction_service = PredictionService()
print("start")
student_score = prediction_service.predict(
    question_id=1,
    comprehension=comprehension,
    question=question,
    question_score=15,
    examiner_answer=examiner_answer,
    student_answer=student_answer
)


print("Score:", student_score)
# print("Percentage:", percent)
