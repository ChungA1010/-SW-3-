from django.urls import path
from . import views

# 앱의 이름을 지정해두면 나중에 관리하기 편합니다.
app_name = 'ai' 

urlpatterns = [
    # http://127.0.0.1:8000/ai/~
    path('predict/', views.analyze_ai, name='analayze_ai'),
    path('feedback/',views.return_feedback, name='return_feedback')
]