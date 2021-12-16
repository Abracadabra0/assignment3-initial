import torch
import torch.nn
from torch.utils.data import DataLoader, random_split
from dataset import BiRdQA
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from models import ERNIEGuesser
from tqdm import tqdm
import os
import argparse
import json


class Trainer:
    def __init__(self,
                 model,
                 train_dataset,
                 test_dataset,
                 batch_size=128,
                 n_epochs=400,
                 lr=3e-5,
                 lr_decay=0.99,
                 print_every=10,
                 test_every=10,
                 save_every=20,
                 name="exp",
                 device=torch.device(0)):
        self.model = model.to(device)
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.step = 0
        self.epoch = 0
        self.print_every = print_every
        self.test_every = test_every
        self.save_every = save_every
        self.best_accuracy = 0

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.loss_fn = torch.nn.NLLLoss()
        self.optimizer = Adam((x for x in model.parameters() if x.requires_grad),
                              lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, lr_decay)
        self.name = name
        self.writer = SummaryWriter(f'runs/{name}')
        os.makedirs(f'checkpoints/{name}', exist_ok=True)

    def train(self):
        self.model.train()
        pbar = tqdm(total=self.n_epochs * len(self.train_dataloader))
        for epoch in range(self.n_epochs):
            all_correct = 0
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                (riddle_input_ids,
                 riddle_attention_mask,
                 options_input_ids,
                 options_attention_mask,
                 label) = batch
                log_prob = self.model(riddle_input_ids,
                                      riddle_attention_mask,
                                      options_input_ids,
                                      options_attention_mask)
                loss = self.loss_fn(log_prob, label)
                loss.backward()
                self.optimizer.step()
                self.step += 1
                pred = torch.argmax(log_prob, dim=1)
                correct = (pred == label).sum()
                all_correct += correct
                pbar.update(1)
                self.writer.add_scalar('train/loss', loss.item(), self.step)
                if self.step % self.print_every == 0:
                    print(f"Step {self.step}: loss = {loss.item()}")
            self.scheduler.step()
            self.epoch += 1
            self.writer.add_scalar('train/accuracy', all_correct / len(self.train_dataloader.dataset), self.epoch)
            if self.epoch % self.test_every == 0:
                self.test()
            if self.epoch % self.save_every == 0:
                self.save_state_dict()
        self.save(f"final.pth")

    def test(self):
        self.model.eval()
        with torch.no_grad():
            all_loss = 0
            all_correct = 0
            for batch in self.test_dataloader:
                (riddle_input_ids,
                 riddle_attention_mask,
                 options_input_ids,
                 options_attention_mask,
                 label) = batch
                log_prob = self.model(riddle_input_ids,
                                      riddle_attention_mask,
                                      options_input_ids,
                                      options_attention_mask)
                loss = self.loss_fn(log_prob, label)
                all_loss += loss.item()
                pred = torch.argmax(log_prob, dim=1)
                correct = (pred == label).sum()
                all_correct += correct
        loss = all_loss / len(self.test_dataloader)
        accuracy = all_correct / len(self.test_dataloader.dataset)
        self.writer.add_scalar('test/loss', loss, self.epoch)
        self.writer.add_scalar('test/accuracy', accuracy, self.epoch)
        print(f'Epoch {self.epoch}: test loss={loss}, test accuracy={accuracy}')
        if accuracy > self.best_accuracy:
            for filename in os.listdir(f'checkpoints/{self.name}'):
                if filename.find('best') != -1:
                    os.remove(os.path.join(f'checkpoints/{self.name}', filename))
            self.best_accuracy = accuracy
            self.save("best(accuracy=%.2f).pth" % accuracy)
        self.model.train()

    def validate(self, dataset):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            all_loss = 0
            all_correct = 0
            for batch in dataloader:
                (riddle_input_ids,
                 riddle_attention_mask,
                 options_input_ids,
                 options_attention_mask,
                 label) = batch
                log_prob = self.model(riddle_input_ids,
                                      riddle_attention_mask,
                                      options_input_ids,
                                      options_attention_mask)
                loss = self.loss_fn(log_prob, label)
                all_loss += loss.item()
                pred = torch.argmax(log_prob, dim=1)
                correct = (pred == label).sum()
                all_correct += correct
        print(f'Loss = {all_loss / len(dataloader)}')
        print(f'Accuracy = {all_correct / len(dataloader.dataset)}')
        self.model.train()

    def save_state_dict(self):
        torch.save(self.model.state_dict(), f"checkpoints/{self.name}/epoch={self.epoch}.pth")

    def save(self, path):
        torch.save(self.model, os.path.join('checkpoints', self.name, path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    device = torch.device(args.device)

    print("Loading datasets...")
    dataset = BiRdQA('all.csv', 'wiki_info_v2.json', n_options=5, device=device)
    train_size = int(len(dataset) * 0.8)
    test_size = int(len(dataset) - train_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print("Building model...")
    model = ERNIEGuesser(**config['model_config'])
    trainer = Trainer(model, train_dataset, test_dataset, **config['trainer_config'], device=device)

    print("Start training")
    trainer.train()
