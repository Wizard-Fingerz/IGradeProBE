from .prediction import PredictionService

student_answer = """
Economic partnerships and trade relations: Nations join international organizations to expand markets, reduce trade barriers, negotiate favorable agreements, and promote foreign investments.

Promotion of Security and stability

Access to international aid and support: States benefit from humanitarian assistance, financial aid, and developmental grants during economic crises, natural disasters, or emergencies.

Promotion of cultural exchange: International platforms encourage people-to-people interaction, festivals, scholarships, and exchange programs that foster mutual understanding.

Influence on global norms and standards: Participation ensures a country’s voice in setting international laws, conventions, and ethical standards, shaping global governance.

# """

# student_answer = """
# Free movement of people and goods – ECOWAS allows citizens to travel and trade easily.
# Collective security – member states unite to fight terrorism and civil wars.
# Boost in regional economy through common market policies.
# Cultural exchange programmes among member nations.
# Support for democratic governance and human rights in the region.
# """


examiner_answer = """
Economic partnerships and trade relations  
ii. Promotion of Security and stability  
iii. Access to international aid and support  
iv. Promotion of cultural exchange  
v. Influence on global norms and standards  
vi. Promotion of sustainable development goals (SDGs)  
vii. Technology transfer and innovation  
viii. Crisis management and humanitarian response  
ix. Strengthening regional cooperation  
x. Attracting diaspora investment  
xi. Enhancing national image and soft power  
xii. Facilitating access to global markets  
xiii. Addressing global challenges collaboratively  
xiv. Supporting gender equality initiatives  
xv. Engaging in peacekeeping and conflict resolution  
xvi. Strengthening bilateral and multilateral relations  
xvii. Encouraging foreign direct investment (FDI)  
xviii. Promoting health initiatives and disease control  
xix. Facilitating educational exchange programmes  
xx. Developing crisis-resilient strategies  
xxi. Promoting human rights  
xxii. Participating in the activities of international organisations  
xxiii. Promoting good governance  
xxiv. Strengthening international legal frameworks  
xxv. Promoting national interest  
xxvi. Protection of territorial integrity/sovereignty  
xxvii. Advertising economic potentials
"""

comprehension = """
Economic partnerships and trade relations
→ Nations join international organizations to expand markets, reduce trade barriers, negotiate favorable agreements, and promote foreign investments.

Promotion of Security and stability
→ Membership allows countries to cooperate on defense pacts, intelligence sharing, and conflict prevention, ensuring collective peace and stability.

Access to international aid and support
→ States benefit from humanitarian assistance, financial aid, and developmental grants during economic crises, natural disasters, or emergencies.

Promotion of cultural exchange
→ International platforms encourage people-to-people interaction, festivals, scholarships, and exchange programs that foster mutual understanding.

Influence on global norms and standards
→ Participation ensures a country’s voice in setting international laws, conventions, and ethical standards, shaping global governance.

Promotion of sustainable development goals (SDGs)
→ Cooperation accelerates efforts in poverty eradication, quality education, clean energy, and climate action.

Technology transfer and innovation
→ States gain access to advanced technologies, research collaborations, and innovation from developed nations.

Crisis management and humanitarian response
→ International organizations provide frameworks for collective disaster response, refugee management, and health emergencies.

Strengthening regional cooperation
→ Countries enhance unity, integration, and solidarity within their region, promoting political and economic collaboration.

Attracting diaspora investment
→ Engagement with international communities strengthens trust, making citizens abroad more willing to invest back home.

Enhancing national image and soft power
→ Active participation builds credibility, global reputation, and influence through diplomacy and cultural diplomacy.

Facilitating access to global markets
→ Membership reduces barriers to entry, increases export opportunities, and integrates economies into global supply chains.

Addressing global challenges collaboratively
→ Issues like climate change, terrorism, pandemics, and cybercrime are tackled better through joint efforts.

Supporting gender equality initiatives
→ Organizations promote inclusive policies, women’s empowerment, and elimination of discrimination.

Engaging in peacekeeping and conflict resolution
→ Member states contribute troops, mediators, and resources to prevent or resolve international conflicts.

Strengthening bilateral and multilateral relations
→ Provides a platform for diplomatic ties and negotiations with other member states.

Encouraging foreign direct investment (FDI)
→ Stability, open markets, and agreements attract international investors.

Promoting health initiatives and disease control
→ Cooperation with organizations like WHO ensures joint responses to epidemics and improved healthcare standards.

Facilitating educational exchange programmes
→ Student scholarships, research networks, and academic mobility programs are promoted.

Developing crisis-resilient strategies
→ Collective experience-sharing strengthens disaster preparedness and resilience planning.

Promoting human rights
→ Countries commit to respecting and upholding human rights instruments and conventions.

Participating in the activities of international organisations
→ Enables countries to contribute to decision-making and benefit from cooperative projects.

Promoting good governance
→ International standards and peer reviews improve accountability, transparency, and democratic practices.

Strengthening international legal frameworks
→ States collaborate in drafting treaties and conventions that regulate global relations.

Promoting national interest
→ Membership provides leverage to pursue political, economic, and security interests.

Protection of territorial integrity/sovereignty
→ International law and collective security arrangements protect states from aggression.

Advertising economic potentials
→ Countries showcase their resources, investment opportunities, and development projects to the global community.



"""


question = "In five key points, fully developed, highlight the extent to which foreign policy plays a critical role in shaping national development outcomes."
# question = "List five extent to which foreign policy plays a critical role in shaping national development outcomes."
# question = "List 5 Artificial Inteligence"

# evaluator = PredictionService()
# score, percent = evaluator.predict(comprehension, question, examiner_answer, student_answer, question_score = 15)

prediction_service = PredictionService()
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
