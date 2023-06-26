from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import *
from src.opt.training_pipeline import train_model


def main():
    fix_random_seed(seed=args.seed)

    # Eneble TensorBoard logs
    writer = SummaryWriter(log_dir='./logs/' +
                           args.dataset + '_' + args.tags +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S"))
    writer.add_text('args', namespace2markdown(args))

    # Train and Save best model
    model = train_model(
        model_name=args.model,
        opt=args.opt,
        dataset=args.dataset,
        writer=writer
    )
    save_model(model)

    writer.close()
    print('\n'+24*'='+' Experiment Ended '+24*'='+'\n')


if __name__ == "__main__":
    main()
