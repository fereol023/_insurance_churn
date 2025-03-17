from pydantic.dataclasses import dataclass
from typing import Dict, List
from dataclasses import field

@dataclass
class ABTest:
    alternate_models: List[str]
    thresholds: List[int]
    max_value_threshold: int=23

@dataclass
class Model:
    process_ids_abtest: Dict[int, ABTest] = field(default_factory=lambda: {})

@dataclass
class Models:
    models: Dict[str, Model]

MODELS = Models(
    models={
        'model_00': Model(),
        'model_01': Model(),
        'model_02': Model(
            process_ids_abtest={
                **dict.fromkeys(
                    [0, 1],
                    ABTest(
                        alternate_models=['model_01', 'model_03'],
                        thresholds=[9, 18],
                        max_value_threshold = 23
                        )
                    )
                }
        ),
        'model_03': Model()
    }
)

MODELS.models

ab_test_value = 58

for i, threshold in enumerate([9, 18][::-1]):
    print(i, threshold)
    if (ab_test_value) % 23 > threshold:
        print(f'{threshold}, model_')
        break