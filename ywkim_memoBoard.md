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
>
> 2019년 10월 17일
> 
- - -
> 1. Video Codec
>   * Prediction 관련...
>       * 다시 생각해봐도 Prediction Block이 아닌 Prediction Mode를 예측하려는 
>일종의 fast Algorithm을 CNN등의 알고리즘으로 개발할려고 한다면, 
>학습 데이터 셋을 만들때 SATD를 Hadamard Transform 등의 Fast한 방법이 아닌 DCT등의
>좀 더 정확한 방법으로 만들어야 할 것 같다. 
>       * 이러한 생각을 하게 된 계기는, Deep Learning은 원래 GT의 성능을 넘을 수 는 없다.
> 따라서 GT의 성능을 높여서 학습을하면 높은 성능의 GT를 학습할 수 있을 것이다.
>
>2. Pytorch
>   * FC에 새로운 정보를 Concat할려고 할때 Best한 방법은 Network의 Input으로 정보 값을 
>주고, forward에서 concat하는 것 같다. 아래 예제를 첨부한다. 이러한 경우 Sequential class를 효율적으로 사용할 수 없는데,
>이를 해결할 방법을 지금 당장은 떠오르지 않는다. 성능에는 차이가 없지만, forward를 좀 더 깔끔하게 구현하면 좋을텐데..
```angular2
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = models.inception_v3(pretrained=False, aux_logits=False)
        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, 20)
        self.fc1 = nn.Linear(20 + 10, 60)
        self.fc2 = nn.Linear(60, 5)
        
    def forward(self, image, data):
        x1 = self.cnn(image)
        x = torch.cat((x1, data), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
>3. 논문
>   * **Do Better ImageNet Models Transfer Better?**
>       * CVPR 2019에서 Google이 발표한 논문 중 하나이다.
>       * 결론을 정리하면 transfer Coding에서 일부 네트워크 가중치를 그대로 쓰는 것은 별로다.
>       * **가장 좋은 방법은 Fine Tuning**인데, 사실 지금까지 내가 Transfer Learning을 사용하지 않은 이유는 Video Codec이 기본적으로 YUV도메인이기 때문이였는데, 
>생각해보니 약간 멍청했었던게, 내가 한 네트워크를 제대로 학습해두고 그것을 Transfer 해서 사용하면 되었다.
>       * 그런데 내 실험은 Input을 바꾸는 경우가 많은데 이 경우에도 Transfer Learning이 유효한지는 논문에 나와있지 않다.
>       * ETRI는 Transfer Learning에 대해 어떻게 생각하는지 궁금하다.
>
>
>
>```
>
> 2019년 10월 25일
> 
- - -

> 1. Channel 개수 C, layer 개수 D와 CNN 네트워크 복잡성에 관하여.
>       * 이미지의 width와 height는 고정이라고 볼 수 있다.
>       * layer당 Input Feature Map이 동일한 Network라고 가정하면 (Output Channel도 당연히 같다.) Channel 개수 C는 전체 연산량에 정비례한다.
>       * layer 개수는 어떨까. 처음과 마지막 layer를 제외하면 layer 개수가 늘어나면 전체 연산량이 정비례 할 것이다.
>       * RDB 등의 Channel Concat layer에서는 어떨까. RDB 내부 layer의 개수를 n, growth_rate = g, input layer의 개수를 il, output layer를 ol이라 하면,
>         il*g + (il+g)*g + (il+2g)*g +  . . . + (il + (n-2)g)*g + (il+(n-1)*g)*ol = ((n-1)*il + (g*(n-1)*n)/2)*g + (il + (n-1)*g)*ol 이므로
>         RDN 내부 growth_rate에 O(n^2), layer 개수 n에 O(n^2)을 가짐.
>
>2. layer 수에 따른 성능 변화 및 Global Residual Connection에 대하여...
>
>       * 현재 실험하는 RDN 구조는 다음과 같다.
>

>```
>
> 2019년 10월 27일
> 
- - -

> 1. RDB 실험결과
>   * Input
>       ![RDN_img](./ImageForGit/RDN%20Network.PNG)
>       * Input 은 동일하게 132x132 크기의 CTU를 네트워크 초기에 Padding 없는
>Convolution을 통해 128x128 Luma로 회귀하는 네트워크 이다.
>
>   * Network
>       ![RDN_img](./ImageForGit/RDB.PNG)
>
>       ![RDN_img](./ImageForGit/RDN%20Network2.PNG)
>       * Network의 구조는 다음과 같다. 가장 유의깊게 봐야 하는 부분은, Network Output이 Residual임에도 불구하고, Network Output을 생성하는 Convolution 이전에 Globally한 Residual connection이 필요하였다.
>       * Globally한  Residual connection이 없으면 성능개선이 전혀 없었다.
>       * Output이 Residual이 아닌 경우에도 성능 개선이 없었다.       
>       * 심지어 RDB는 내부에도 Residual connection이 필요하다.
>       * Padding을 날리기 위한 3x3 Conv layer 2개를 1개의 5x5 Conv를 Padding 없이 사용하여 보았을때도 성능저하가 극심하였다.
>   * 처음에는 비교적 깊은 네트워크를 차용하였다.
>   1. 일반 Convolution featuermap : 32, RDN 갯수 : 3, RDN 내부 layer 개수 : 6, growth rate : 16
>       * 평균 0.15dB PSNR gain
>   2. 일반 Convolution featuermap : 32, RDN 갯수 : 4, RDN 내부 layer 개수 : 6, growth rate : 24
>       * 평균 0.20dB PSNR gain
>