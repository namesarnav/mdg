#### Dataset creation scripts/notebooks for temporal reasoning capabilities
- [x] [Time expression recognition](https://huggingface.co/datasets/nlpatunt/timex-recognition)
- [ ] Time expression grounding

### Timne expression recognition

The dataset is available at [this link](https://huggingface.co/datasets/nlpatunt/timex-recognition). The task is to given a document, identify all time expressions in it, and their type. The types are `DATE`, `TIME`, `DURATION`, `SET`. The dataset is created using TempEval-3 and TimeBank.

### Time expression grounding

The dataset is available at [this link](https://huggingface.co/datasets/nlpatunt/timex-grounding). The input **context** is a [document](datamodels.py) and the query is a single [time expression](datamodels.py). The task is to a) find an anchor time, i.e., which timex (id) the expression can be anchored with b) once the anchor time is found, ground the text in the query timex to a UTC value (when possible). 

See more temporal capabilities [at this link](https://docs.google.com/spreadsheets/d/1GKqU1AB97O7bckio3-zvtQWiCN8KKRI2SBkpy0MlfY4/edit?gid=0#gid=0)
