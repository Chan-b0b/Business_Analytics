이번 튜토리얼에서는 semi-supervised learning의 기법 중 하나인 **Dual Student**을 강화학습 모델 중 Policy Gradient 모델에서 사용되는 **Baseline**
에 적용시켜 그 효과를 확인하는 것을 목표로 한다. 

### 1. 문제 정의 (TSP with Reinforcement Learning)
![image](https://user-images.githubusercontent.com/93261025/209648923-9b85d092-40da-4ffd-8ab7-618b1b6d2aa3.png)
TSP, 우리말로 외판원 문제는 가장 대표적인 조합 최적화 문제 중 하나이다. 문제 자체는 한 외판원이 N개의 도시를 한 번씩 방문할 때
최소 길이를 같는 경로를 찾는 것으로 간단하나, NP-Hard 집합에 속한다.

최근 이 TSP 문제를 강화학습으로 풀고자 하는 시도들이 진행되고 있다. 우선 Pointer Network(Vinyals, 2015)의 구조를 토대로 
Neural Combinatorial Optimization with Reinforcement Learning(Bello, 2017)에서 REINFORCE 기법을 활용하여
처음으로 TSP 문제를 강화학습으로 풀 수 있다는 가능성을 보였다. 그 이후에 Attention 모델이 주목을 받기 시작하면서 이 분야에서도 
ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!(Kool, 2019)이 attention 모델을 사용함으로서
노드가 20개인 TSP20 문제에서 최적값과 차이가 0.08%까지 줄어드는 결과를 보여줬다. 해당 모델은 아래 그림처럼 Query 값을 현재 노드, 경로 시작 노드,
그리고 노드 전체 정보 전부를 고려한 값으로 지정하여 아직까지 선택되지 않은 노드들과의 attention score를 매겨 softmax한 값을 확률로
선택해나아가게끔 설계되어있다. 노드들이 모두 선택된 이후에는 해당 경로의 길이를 구해 그 음수값을 return으로 사용한다.

![image](https://user-images.githubusercontent.com/93261025/209677644-1d20fb47-0d83-422f-94a6-575a71b58926.png)

하지만 Kool은 본인 논문의 contribution으로 attention 모델을 적용한 것 외에도 **baseline을 어떻게 설정했는지** 또한 강조했다. 

![image](https://user-images.githubusercontent.com/93261025/209675797-bedf4101-7672-471f-bdf8-020887b8c88a.png)

#### Baseline
Baseline이란 Policy Gradient 모델의 문제점 중 하나인 큰 variance를 줄이기 위한 기법으로, 현재 policy &#920; 와 독립적인 값 baseline을 
뺌으로서 적용한다. Policy gradient는 해당값의 미분값을 사용하기 때문에 이렇게 임의의 baseline을 빼는 것은 bias를 생성하지 않는다. 
이렇게 사용하는 baseline은 보통 실제 return값의 추정값을 사용하게 되는데, 강화학습 기법 중 많이 사용되는 기법 중 하나인 actor-critic의 경우
별도의 critic network를 만들어 state를 input으로 받고 이 baseline값을 추정하게끔 학습을 진행한다. 
하지만 Kool은 이렇게 별도의 모델을 만드는 것은 추가적인 연산을 필요하여 비효율적이므로, return값을 확률기반 sampling으로 진행하는 것과 달리
argmax값만을 선택하는 greedy rollout을 통해 도출하는 방법을 제시한다. 이는 critic network를 생략할 수 있고, 결과적으로 수렴 속도도 더
빠르다고 한다. 이에 더 나아가 POMO(Kwon, 2021)에서는 어떠한 노드에서 시작하더라도 모든 경우에 대해서 같은 경로를 그려야한다는 TSP의 
속성을 이용하여 이 baseline 값을 모든 노드에서 시작했을 때 나온 각 경로의 평균값으로 설정했다. 이 경우 greedy rollout하는 시간도 
줄일 수 있는 동시에 그 안정성도 향상되어 학습이 빠르게 진행되었다고 한다.

### 2. Motivation
POMO는 별도의 critic 모델의 필요성도 없앴고, rollout을 추가적으로 해야하는 번거로움도 없앴다는 점에서 시사하는 바가 크다고 생각된다.
다만, 결국에는 하나의 모델에서 나온 결과값들을 baseline으로 계속해서 사용하기 때문에 
**labeled data가 없는 상황에서 unstable predictions에 의한 문제**
가 생길 수 있겠다는 생각이 들었다. 이는 Dual Student에서 제기했던 문제점과 유사하다고 판단하여, 
이 문제 상황에 해당 기법을 한 번 적용해보고자 한다. 이 문제는 강화학습의 exploration을 통해 해결될 수 있다고
생각할 수 있으나, 아래 그림과 같이 200 epoch 정도에서는 policy 개선 속도가 현저하게 줄었음에도 결국 학습은 2000 epoch까지 
시켜야하는 상황을 고려해봤을 때, 현재와 같은 exploration 기법에만 의존하기에는 너무 비효율적이라는 판단을 내렸다.

![image](https://user-images.githubusercontent.com/93261025/209679140-a18b115b-2124-4e2a-9bff-e4d024148a54.png)

### 3. Experiments
실험은 다음과 같이 진행했다. 우선 현재까지 가장 좋은 성능을 보이고 있는 POMO 모델을 기반으로 한다.

```
    self.encoder = TSP_Encoder(**model_params)
    self.decoder_student = TSP_Decoder(**model_params)
    self.decoder_student1 = TSP_Decoder(**model_params)  
```
TSP 노드들의 embedding을 산출하는 encoder는 하나를 공용으로 설정하고, 두 개의 decoder가 분개하게 만들어 두 student로써 작동하게 한다. 

```
    # Max_Student (= Dual Student)
    mean_advantage = torch.amax(torch.stack([reward1,reward2], dim=0),dim=0)            
    # Average_Student
    mean_advantage = torch.mean(torch.stack([reward1,reward2], dim=0),dim=0)
    # Average_Node
    mean_advantage = torch.mean(torch.mean(torch.stack([reward1,reward2], dim=0),dim=2,keepdim=True),dim=0)
        
```

Baseline값을 어떻게 설정하느냐에 따라 달라지는 모델 성능을 평가하게 되는데, 첫 번째로 dual student와 가장 유사하게끔 두 모델 중 더 
짧은 경로를 가지는 값을 baseline 값으로 설정하는 max_student가 있다. 그 다음으로는 두 student의 평균값을 baseline으로 설정하는 두 방법이 
있는데, 하나는 노드별로 평균을 따로 계산하는 average_student와 모든 노드들의 평균값을 한 번에 계산하는 average_node가 있다. 

```
trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 25,
    'train_episodes': 100 * 1000,
    'train_batch_size': 80,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_tsp20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 510,  # epoch version of pre-trained model to laod.

    }
}
```
각 실험은 한 epoch 당 random generate된 10만개의 TSP 문제를 풀게 되고, 시간 관계상 25 epoch만 진행했다 (i7‑7‑11850HE, RTX-3090Ti 기준 한 epoch 당 4시간)

### 4. Results

![image](https://user-images.githubusercontent.com/93261025/209683278-7849ff79-4dc6-48fb-801c-678fda5711c0.png)

짧은 경로를 가지는 값을 baseline 값으로 설정하는 max_student가 있다. 그 다음으로는 두 student의 평균값을 baseline으로 설정하는 두 방법이 
있는데, 하나는 노드별로 평균을 따로 계산하는 average_student와 모든 노드들의 평균값을 한 번에 계산하는 average_node가 있다. 
|  Baseline  |   Path Length |
|:--:|:--------:|
|  single |        8.043 | 
|  max_student |        8.208 | 
|  average_student |        8.039 | 
|  average_node |        8.025 |

실험 결과 한 개의 decoder만을 가지고 실험했을 때보다 두 개의 decoder를 가지고 average_node 기법을 사용했을 때 smoothing 값 기준으로
각각 8.043, 8.025로 0.22%의 성능 향상이 이뤄진 것을 볼 수 있었다. 이는 현재 2000 epoch까지 학습된 POMO 모델이 최적값과 1.07% 차이를 
보이는 점을 감안하면 결코 작은 차이가 아님을 알 수 있다. 그 밖에도 average_student 기법 또한 8.039로 single 모델 보다는 좋은 성능을 보임을 확인할 수 있으나, max_student의 경우 한참 떨어지는 결과를 보였다. 이 이유는 아래 그래프로 설명이 가능한데,

![image](https://user-images.githubusercontent.com/93261025/209684582-9ecce46f-018b-488b-9dc6-e11626840f4a.png)

해당 그래프는 max_student와 average_node의 loss 값 추이를 나타낸다. 앞서 Baseline을 사용하는 가장 큰 이유 중 하나가 큰 variance를 줄이기 위함이라고 언급했었다. 위 그래프처럼 계속해서 큰 variance를 가져가는 경우 학습이 더디게 진행될 수 밖에 없고, 이는 낮은 return 값으로 이어진 것으로 보인다. 

### 5. Discussion
위 결과를 통해 같은 양의 데이터로 학습을 진행했을 때 Dual Student 기법에서 착안한 baseline을 사용시 성능이 향상된다는 것을 알 수 있었다. 하지만 해당 방법의 경우 연산을 두 배로 진행함에 따라 소요 시간이 증가한다는 문제점이 있다. 아래 그림은 시간을 x축으로 한 그래프이다. 

![image](https://user-images.githubusercontent.com/93261025/209685455-ef431ac2-92de-4290-a797-b03f4cb770bb.png)

Batch_step을 x축으로 했을 때는 안보이던 해당 방법론의 문제가 두드러진다. Single model의 경우 2시간 안쪽으로 25 epoch을 끝내게 되면서 다른 모델 대비 시간을 기준으로 했을 때 훨씬 더 빠르게 해를 찾는 것을 확인할 수 있다. 해당 결과를 봤을 때는 두 배 더 많은 데이터로 학습을 진행할 수 있는 single model이 우위에 있다고 생각이 들 수 있다. 하지만 학습 속도가 빠르게 진행되는 초반과 달리 후반으로 갈수록 gradient는 0에 수렴하게 되면서 학습 속도가 더디게 된다. 더 오랜 시간동안 학습을 진행하였을 경우에는, 같은 시간대에서도 이번 튜토리얼에서 제시하는 방법론의 성능이 더 우월할 수 있다는 가능성을 제시하고 싶다. 이는 후속 연구에서 밝혀보고자 한다.

#### References
- Bello, Irwan, et al. "Neural combinatorial optimization with reinforcement learning." arXiv preprint arXiv:1611.09940 (2016).
- Nazari, Mohammadreza, et al. "Reinforcement learning for solving the vehicle routing problem." Advances in neural information processing systems 31 (2018).
- Kool, Wouter, Herke Van Hoof, and Max Welling. "Attention, learn to solve routing problems!." arXiv preprint arXiv:1803.08475 (2018).
- Kwon, Yeong-Dae, et al. "Pomo: Policy optimization with multiple optima for reinforcement learning." Advances in Neural Information Processing Systems 33 (2020): 21188-21198.

