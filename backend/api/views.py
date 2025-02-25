from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from ai_engine.inference import get_chatbot_response

@api_view(['GET'])
def home(request):
    return Response({"message": "Welcome to Servia AI Backend!"})


@api_view(['GET'])
def chatbot_response(request):
    user_message = request.GET.get("message", "")
    if not user_message:
        return Response({"error": "Message cannot be empty"}, status=400)

    response = get_chatbot_response(user_message)
    return Response({"response": response})
