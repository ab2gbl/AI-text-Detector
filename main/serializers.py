
from rest_framework import serializers

class TextSerializer(serializers.Serializer):
	text = serializers.CharField(max_length=5000000)