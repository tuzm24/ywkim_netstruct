*딥러닝 실험 노트*
=============
>
> 2019년 10월 16일
> 
- - -
1. pytorch
    * 각각의 배치마다 gpu에 네트워크를 gpu에 올린다. 즉 batchsize*netparameters가 gpu 위에 올라가는 듯 하다.
        *실험 예제
    * 실험 초기에 네트워크가 학습을 포기한다면 네트워크의 깊이나 넓이를 넓혀 보라.
```
# 배치사이즈가 100이여도 전체 메모리의 20% 밖에 차지 하지 않음.
Total params: 57,186
Trainable params: 57,186
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.27
Forward/backward pass size (MB): 84.75
Params size (MB): 0.22
Estimated Total Size (MB): 85.24
----------------------------------------------------------------

================================================================
# 배치사이즈가 49이여도 전체 메모리의 80% 차지.
Total params: 64,872
Trainable params: 64,872
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.27
Forward/backward pass size (MB): 747.50
Params size (MB): 0.25
Estimated Total Size (MB): 748.02
----------------------------------------------------------------
```

2. 

