from transformers import TrainingArguments
import inspect

# Print the signature of TrainingArguments
print("TrainingArguments signature:")
print(inspect.signature(TrainingArguments))

# Print the full docstring (optional, gives default values)
print("\nTrainingArguments docstring:")
print(TrainingArguments.__doc__)

# Print all attributes to see what arguments might exist
print("\nAll attributes of TrainingArguments class:")
print([attr for attr in dir(TrainingArguments) if not attr.startswith("_")])