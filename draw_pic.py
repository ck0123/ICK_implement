#print the different
import torch

# checkpoint = torch.load('./drive/checkpoint/ckpt_resnet18_90.t7')
acc_list = []
x_list = [i for i in range(10,110,10)]
for i in x_list:
  checkpoint = torch.load('./drive/My Drive/checkpoint/ckpt_resnet18_'+str(i)+'.t7')
  acc_list.append(checkpoint['acc'])

acc_frozen_list = []
x2_list = [i for i in range(20,120,10)]
for i in x2_list:
  checkpoint = torch.load('./drive/My Drive/checkpoint/ckpt_resnet18_'+str(i)+'_frozen.t7')
  acc_frozen_list.append(checkpoint['acc'])


import matplotlib.pyplot as plt
plt.plot(x_list,acc_list,label="NORMAL")
plt.plot(x2_list,acc_frozen_list,label="ICK")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("Show the result,acc of test")
plt.legend()
plt.show()


#loss data

loss_test_list = [0.665,0.590,0.440,0.508,0.496,0.528,0.439,0.475,0.437,0.466]
loss_train_list = [0.478,0.387,0.355,0.337,0.334,0.323,0.319,0.310,0.307,0.312]
loss_train_frozen_list = [0.389,0.305,0.274,0.275,0.261,0.257,0.249,0.239,0.252,0.244]
loss_test_frozen_list  = [0.418,0.363,0.332,0.332,0.328,0.306,0.311,0.306,0.326,0.312]


plt.plot(x_list,loss_train_list,label="NORMAL")
plt.plot(x2_list,loss_train_frozen_list,label="ICK")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Show the result,loss of train")
plt.legend()
plt.show()