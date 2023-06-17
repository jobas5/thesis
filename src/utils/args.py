import torch
import argparse


def print_args(ARGS):
    print('\n'+26*'='+' Configuration '+26*'=')
    for name, var in vars(ARGS).items():
        print('{} : {}'.format(name, var))
    print('\n'+25*'='+' Training Starts '+25*'='+'\n')



def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')

    PARSER.add_argument('--dataset', default='iris', type=str, 
                        choices=['iris', 'wine'], help='Dataset.')
    
    PARSER.add_argument('--model', default='densenet121', type=str, 
                        choices=['resnet18', 'densenet121'], help='Model.')
    
    PARSER.add_argument('--opt', default='Adagrad', type=str, 
                        choices=['Adam', 'Adamax', 'Adagrad'], help='Optimizer.')
    

    PARSER.add_argument('--epochs', default=10, type=int, 
                        help='Number of Training Epochs.')
    
    PARSER.add_argument('--batch_size', default=16, type=int, 
                        help='Batch Size.')
    
    PARSER.add_argument('--val_size', default=0.2, type=int, 
                        help='Validation Size. Propotion of the training Dataset.')
    
    PARSER.add_argument('--test_size', default=0.2, type=int, 
                        help='Test Size. Propotion of Test Dataset.')
    

    PARSER.add_argument('--seed', default=0, type=int, 
                        help='Fix Random Seed.')
    
    PARSER.add_argument('--tags', default='logs', type=str, 
                        help='Run Tags.')
    PARSER.add_argument('--device', default=None, type=str,
                        help='Device to run Experiment')
    

    ARGS = PARSER.parse_args()

    if ARGS.device is None:
        ARGS.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print_args(ARGS)
    return ARGS

args = parser()

if __name__ == "__main__":
    pass