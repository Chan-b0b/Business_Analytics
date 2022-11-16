# Autoencoder

<p align="center">
    <img src="images/autoencoder_architecture.png" width="300"/>
</p

그동안 이미지 관련 문제에서 Anomaly Detection이 필요할 때마다 많이 써왔던 Autoencoder인데,\
이번 수업을 계기로 그동안 가져왔던 의문점을 풀어보면 좋겠다는 생각에 이번 튜토리얼 주제로 결정했다. 


# 1_Dataset
우선 데이터는 특성이 명확하면서 다루기 쉬운 MNIST 데이터로 결정했다. 

```Python
from torchvision.datasets import MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (1.0,))
])
download_root = './MNIST_DATASET'
train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
```
위의 코드를 사용하면 웹상에서 torch dataset 형태로 손쉽게 데이터셋을 불러올 수 있다.

```Python
batch_size = 128 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```
torch에서 제공하는 DataLoader를 사용하면 batch_size, transform, shuffle, multi-gpu 등 다양한 조건들을 손쉽게 세팅할 수 있기 때문에 모델 학습시 즐겨 사용한다.

# 2_Latent Dimension

많은 textbook이나 reference에서 상응하는 결과를 보여왔으므로
naive한 생각으로 당연히 latent dimension이 증가할수록 output되는 결과가 더 좋아질 것으로 보인다.
그런데 내가 궁금한 점은, 우리는 latent feature 안에 노이즈 및 불필요한 정보가 제외된, 중요한 정보만이 남겨있다는 기대(?)를 하고 있는데,
과연 육안으로 우리가 그 것을 확인할 수 있을까 하는 것이다. \
따라서 Latent Dimension을 중가시키면서 output 되는 이미지들을 확인해보고자 했다.

```Python
num_epochs = 50
for dim in range(1,11):
   train_dataset, test_dataset, train_loader, test_loader = create_dataset()
   loss_fn, model, optim = init_model(latent_dim = dim)
   min_loss = np.inf
   diz_loss = {'train_loss':[],'val_loss':[]}
   for epoch in range(num_epochs):
      train_loss = train_epoch(model,device,train_loader,loss_fn,optim)
      val_loss = test_epoch(model,device,test_loader,loss_fn)
      
      if val_loss < min_loss :
         opt_model = model
         best_epoch = epoch
      diz_loss['train_loss'].append(train_loss)
      diz_loss['val_loss'].append(val_loss)
   print(f"epoch : {best_epoch}, Val_loss : {np.min(diz_loss['val_loss'])}")
   save_dict = {
      "model" : opt_model.state_dict(),
      "loss" : diz_loss
   }
   torch.save(save_dict, f'model/number_{number}')
   plot_ae_outputs(opt_model,test_dataset,n=10,name='unpooling', num=dim)
```

<p align="center">
    <img src="images/latent.png" width="600"/>
</p
<p align="center">
    <img src="images/latent2.png" width="600"/>
</p
<p align="center">
    <img src="images/latent5.png" width="600"/>
</p
<p align="center">
    <img src="images/latent10.png" width="600"/>
</p




