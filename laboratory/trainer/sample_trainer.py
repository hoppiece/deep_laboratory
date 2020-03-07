from torch import nn
from torch import optim
from torch.autograd import Variable


EPOCH_NUM = 2

class SampleTrainer():
    def __init__(self, net, train_data):
        self.dset_type = {"mnist"}
        self.net = net
        self.traindata = train_data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr = 0.001,
            momentum = 0.9
            )

    def training(self):
        print("Start training.")

        for epoch in range(EPOCH_NUM):
            running_loss = 0.0

            for i, data in enumerate(self.traindata):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print("[{}, {:>5}] loss: {:.3f}".format(
                        epoch + 1,
                        i+1,
                        running_loss / 100))
                running_loss = 0.0

        print("Finished Training")
                
