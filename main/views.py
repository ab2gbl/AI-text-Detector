from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import TextSerializer
import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from .prediction import predict_article
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "ai_detection_model_200k")

# Load the model and tokenizer once
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to(device)  # Move model to the appropriate device





# API View
class Detect(APIView):
    def post(self, request):        
        serializer = TextSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.data['text']
            
            try:
                # Use the predict_article function
                label, ai_count, human_count = predict_article(text, model, tokenizer)
                
                response_data = {
                    "text": text,
                    "label": label,
                    "ai_count": ai_count,
                    "human_count": human_count,
                }
                return Response(response_data, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        

def Page(request):
    return render(request,'main.html') 		