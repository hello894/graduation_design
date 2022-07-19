import torch
import utils
import os
import numpy as np
from pathlib import Path
import send_msg
import time
from torchsummary import summary


respath = "D:\\graduation design\\SIMCLR-pytorch\\simclr\\simclr\\results\\hpatches"


class SimCLR:
    def __init__(self, model, optimizer, dataloaders, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn
        
    def load_model(self, args):
        self.model.load_state_dict(torch.load(args.model_path), strict=False)

        if 'remove_top_layers' in vars(args):
            if args.remove_top_layers > 0:
                if args.multiple_gpus:
                    temp = list(self.model.module.fc.children())
                    if args.remove_top_layers <= len(temp):
                        self.model.module.fc = torch.nn.Sequential(*temp[:-args.remove_top_layers])
                else:
                    for i in range(5):
                        temp = list(self.model.fc.children())
                        if args.remove_top_layers <= len(temp):
                            self.model.fc = torch.nn.Sequential(*temp[:-args.remove_top_layers])
                    #summary(self.model,(3,65,65))



    def get_representations(self, args, mode):

        self.model.eval()

        res = {
        'X':torch.FloatTensor()
        }

        with torch.no_grad():
            for batch in self.dataloaders[mode]:

                x = batch['image'].to(args.device)


                # get their outputs
                pred = self.model(x)

                res['X'] = torch.cat((res['X'], pred.cpu()))



        res['X'] = np.array(res['X'])


        return res

    def train(self, args, num_epochs, log_interval):
        '''
        trains self.model on the train dataset for num_epochs
        and saves model and loss graph after log_interval
        number of epochs
        '''

        batch_losses = []

        def logging():
            # Plot the training losses Graph and save it

            Path(os.path.join(respath, "plots")).mkdir(parents=True, exist_ok=True)

            utils.plotfuncs.plot_losses(batch_losses, 'Training Losses',
                                        os.path.join(respath, 'plots/training_losses-bs128-res18_hpatches.png'))

            Path(os.path.join(respath, "model")).mkdir(parents=True, exist_ok=True)

            # Store model and optimizer files
            torch.save(self.model.state_dict(), os.path.join(respath, "model/model.pth"))
            torch.save(self.optimizer.state_dict(), os.path.join(respath, "model/optimizer.pth"))
            np.savez(os.path.join(respath, "model/lossesfile"), np.array(batch_losses))

        self.model.train()

        # run a for loop for num_epochs
        for epoch in range(num_epochs):
            print("epoch:" + str(epoch))
            time_tuple = time.localtime(time.time())
            print("当前时间为{}年{}月{}日{}点{}分{}秒".format(time_tuple[0], time_tuple[1], time_tuple[2], time_tuple[3],
                                                   time_tuple[4], time_tuple[5]))
            # run a for loop for each batch
            for batch in self.dataloaders['train']:
                # zero out grads
                self.optimizer.zero_grad()

                x1 = batch['image1'].to(args.device)

                x2 = batch['image2'].to(args.device)

                # get their outputs
                y1 = self.model(x1)
                y2 = self.model(x2)

                # get loss value
                loss = self.loss_fn(y1, y2)

                batch_losses.append(loss.cpu().data.item())

                # perform backprop on loss value to get gradient values
                loss.backward()

                # run the optimizer
                self.optimizer.step()

            if epoch % log_interval == log_interval - 1:
                logging()

        logging()

