# WebAgentGrounding

## Running the baseline
Run the baseline with
```
python weblinx_baseline/main.py
```

## Design your grounding model
To design your own grounding model, use the template in model.py. Then you can evaluate it like this:
```python
from eval import evaluate

my_model = MyModel()
evaluate(my_model, split='testing', model_name='my_model')
```