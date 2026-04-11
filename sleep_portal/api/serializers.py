from rest_framework import serializers


class PredictRequestSerializer(serializers.Serializer):
    # features: list of float values (một epoch)
    features = serializers.ListField(
        child=serializers.ListField(
            child=serializers.FloatField(),
            min_length=1,
        ),
        min_length=1,
        max_length=10,  # max 10 epochs per request
    )
