*딥러닝 아이디어 노트*
=============

>
>
>## In-loop filter 관련
>>### 2019년 10월 16일 
> - - -
>### 1. SENet의 NN에 QP 추가 하여 Feature Map 재조정
>   * SENet의 모듈을 이용하여 QP나 ALF Mode Index등의 CTU 전체에 같은 값이 적용되는값에 대하여 Complex측면이나 효율성에서 더 나은 정보를 전할 수 있을 것으로 보임.
>   * 많은 Convolution layer중에 어디에 NN QP를 추가 할지는 여러번의 실험으로 결정해야 할것 (네트워크 초반, 중반, 후반, 모두 등등..)
>   
>
>       + 현재 MobileNetwork의 Anchor가 실험 완료되는대로 순차적 실험 예정. 
>
>### 2. Unfiltered Reconstruction을 활용한 Encoder-Decoder Side Prediction Filtering
>   * 
