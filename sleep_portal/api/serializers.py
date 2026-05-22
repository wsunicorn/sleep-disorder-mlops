from rest_framework import serializers


def _get_feature_count() -> int:
    try:
        from inference.predictor import get_feature_count
        return get_feature_count()
    except Exception:
        return 24  # Kaggle CAP notebook feature schema


class PredictRequestSerializer(serializers.Serializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = _get_feature_count()
        self.fields["features"] = serializers.ListField(
            child=serializers.ListField(
                child=serializers.FloatField(),
                min_length=n,
                max_length=n,
            ),
            min_length=1,
            max_length=256,
        )
