# Knowledge Distillation

## References
* Technical Report: http://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf
* Github: https://github.com/peterliht/knowledge-distillation-pytorch

## Hyperparameters
1. Temperature scale (T):
```bash
Z = f(X)
O = softmax(Z / T)
```
2. Alpha
```bash
given: data - X, label - Y
---------------------------------------------------------
Z_teacher = teacher(X)
Z_student = student(X)
O_teacher = softmax(Z_teacher / T)
O_student = softmax(Z_student / T)
---------------------------------------------------------
# O_teacher * log(O_student)
loss1 = CrossEntropy(O_student, O_teacher)
# Y * log(O_student)
loss2 = CrossEntropy(softmax(Z_student / 1), Y)
---------------------------------------------------------
loss = alpha * (T ** 2) * loss1 + (1 - alpha) * loss2
```

## Usage
1. Train Teacher

```bash
python teacher_train.py --batch-size 64 --lr 0.01 --num-epochs 10 --device 'cuda:0'

```

2. Train Student

```bash

```
