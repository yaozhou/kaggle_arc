import torch
import torch.nn as nn
import torch.nn.functional as F

from config import gamma
class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, cartpole_test, evaluation_mode):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.evaluation_mode = evaluation_mode

        self.fc1 = nn.Linear(num_inputs, 10 if cartpole_test else 32 * 80)
        self.fc2 = nn.Linear(10 if cartpole_test else 32 * 80, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, device):
        states = torch.stack([torch.Tensor(s).to(device) for s in batch.state])
        next_states = torch.stack([torch.Tensor(s).to(device) for s in batch.next_state])
        actions = torch.Tensor(  batch.action ).float().to(device)
        rewards = torch.Tensor(batch.reward).to(device)
        masks = torch.Tensor(batch.mask).to(device)

        pred = online_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        if (self.evaluation_mode):
            print(qvalue)
        _, action = torch.max(qvalue, 1)
        return action.cpu().numpy()[0]
