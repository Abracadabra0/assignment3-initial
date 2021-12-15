import torch
import torch.nn
from torch.utils.data import DataLoader, random_split
from dataset import BiRdQA
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.tensorboard import SummaryWriter
from models import ERNIEGuesser


class Trainer:
    def __init__(self,
                 model,
                 train_dataset,
                 test_dataset,
                 batch_size=64,
                 n_epochs=200,
                 lr=1e-4,
                 lr_decay=0.98,
                 print_every=10,
                 test_every=10,
                 save_every=10,
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

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.loss_fn = torch.nn.NLLLoss()
        self.optimizer = Adam((x for x in model.parameters() if x.requires_grad),
                              lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, lr_decay)
        # self.writer = SummaryWriter()

    def train(self):
        self.model.to(self.device)
        self.model.eval()
        for epoch in range(self.n_epochs):
            print(f"----------------Epoch {epoch}----------------")
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
                # self.writer.add_scalar('train/loss', loss.item(), self.step)
                if self.step % self.print_every == 0:
                    print(f"Step {self.step}: loss = {loss.item()}")
            self.scheduler.step()
            self.epoch += 1
            if self.epoch % self.test_every == 0:
                self.test()
            if self.epoch % self.save_every == 0:
                self.save_state_dict()
        self.save()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            all_loss = 0
            all_correct = 0
            i_iter = 0
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
                i_iter += 1
            print(f'Test Loss = {all_loss / i_iter}')
            print(f'Test Accuracy = {all_correct / len(self.test_dataloader)}')
        # self.writer.add_scalar('test/loss', all_loss / i_iter, self.epoch)
        # self.writer.add_scalar('test/accuracy', all_correct / len(self.test_dataloader), self.epoch)
        self.model.train()

    def validate(self, dataset):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        with torch.no_grad():
            all_loss = 0
            all_correct = 0
            i_iter = 0
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
                i_iter += 1
        print(f'Loss = {all_loss / i_iter}')
        print(f'Accuracy = {all_correct / len(dataloader)}')
        self.model.train()

    def save_state_dict(self):
        torch.save(self.model.state_dict(), f"checkpoints/epoch={self.epoch}.pth")

    def save(self):
        torch.save(self.model, f"checkpoints/final.pth")


if __name__ == '__main__':
    device = torch.device(0)
    train_dataset = BiRdQA('train.csv', 'wiki_info_v2.json', n_options=5, device=device)
    val_dataset = BiRdQA('val.csv', 'wiki_info_v2.json', n_options=5, device=device)
    train_size = int(len(train_dataset) * 0.8)
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    model = ERNIEGuesser(n_options=5, n_layers=6, n_hidden=512, feature_concat='all', use_ln=True, dropout=0.1)
    trainer = Trainer(model, train_dataset, test_dataset, device=device)
    trainer.train()
    trainer.validate(val_dataset)
