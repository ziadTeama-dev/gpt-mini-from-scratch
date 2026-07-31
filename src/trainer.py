import torch
import math
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        lr=3e-4,
        std_noise= .2,
        epochs=10,
        device="cuda",
        save_path="gpt_mini.pth",
        Logs_path="Logs/test"
    ):

        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.std_noise = std_noise

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(log_dir=Logs_path)



    def train(self ):

        self.model.train()
        global_step = 0
        try:
            for epoch in range(self.epochs):
    
                total_loss = 0
    
                progress = tqdm(
                    self.dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}"
                )
    
    
                for x,y in (progress):
                    global_step +=1
    
                    x = x.to(self.device)
                    y = y.to(self.device)
    
    
                    # forward  
                    # input , mask , padding mask , std_noise
                    logits, _ = self.model( x, True, None, self.std_noise)
    
    
                    B,S,V = logits.shape
    
    
                    # CrossEntropy expects:
                    # (N,C) , (N)
                    loss = self.loss_fn(
                        logits.reshape(B*S,V),
                        y.reshape(B*S)
                    )
    
    
    
                    # backward
                    self.optimizer.zero_grad()
    
                    loss.backward()

                    # record the gradiant
                    total_grad=0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            total_grad+= param.grad.norm(p=2).item()**2

                    total_grad = math.sqrt(total_grad)

                    # record loss each batch
                    self.writer.add_scalar("Loss per batch" ,
                                           loss.item(),
                                           global_step)
                    # as well as loss
                    self.writer.add_scalar("grad_norm",
                                        scalar_value=total_grad,
                                        global_step=global_step)



    
    
                    # prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        1.0
                    )
    
    
                    self.optimizer.step()
    
    
                    total_loss += loss.item()
    
    
                    progress.set_postfix(
                        loss=loss.item()
                    )
    
    
                avg_loss = total_loss / len(self.dataloader)
    
                self.writer.add_scalar(tag="Loss",
                                       scalar_value=avg_loss,
                                       global_step=epoch)
    
    
                print(
                    f"Epoch {epoch+1} Loss: {avg_loss:.4f}"
                )
    
    
                self.save()

        except KeyboardInterrupt :

            print("The user has stoped the training..")
            self.writer.close()

        finally:
            self.writer.close()
    


    def save(self):

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            },
            self.save_path
        )


    def load(self,path):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )


        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )