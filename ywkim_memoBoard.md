*딥러닝 실험 노트*
=============
>
> 2019년 10월 16일
> 
- - -
> 1. pytorch
>   * 각각의 배치마다 gpu에 네트워크를 gpu에 올린다. 즉 batchsize*netparameters가 gpu 위에 올라가는 듯 하다. (아래 예제 참조)
>   * 실험 초기에 네트워크가 학습을 포기한다면 네트워크의 깊이나 넓이를 넓혀 보라.
>

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
> 2. Video Codec
> * Prediction 관련...
>   * Prediction에서 만큼은 Pixel의 L1, L2 loss를 사용하면 안 될 것 같다.
>   * Inter는 잘 모르겠지만 Intra에서 Prediction과 Original의 SATD를 계산한다.
>   * 그 이유는 Residual의 Transform Domain에서의 신호 몰림을 최대화 하기위해서로 추정된다. (아래 예제)
```----------------------------------------------------------------
예를들어 Original 신호가 500 500 500 이라면 
Prediction을 100 100 100으로 하든 450 450 450 으로 하든 
Residual의 Transform 신호는 DC에 몰리게 되고, 
이를 Quantization을 실행하면 에러는 확률적으로 비슷하다.
하지만 Pixel Domain에서의 MSE는 매우 차이가 크다.
----------------------------------------------------------------
```
