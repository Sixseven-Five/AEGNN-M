from argparse import ArgumentParser

def add_train_argument(p):
    p.add_argument('--dataset_name', type=str,default='bbbp.csv',
                   help='The name of input CSV file.')
    p.add_argument('--data_dir',type=str,default='./data/MoleculeNet/',
                   help='The dir of input CSV file.')
    p.add_argument('--save_dir',type=str,default='./result/model/',
                   help='The dir to save output model.pt.,default is "result/model/"')
    p.add_argument('--log_dir',type=str,default='./result/log/',
                   help='The dir of output log file.')
    p.add_argument('--dataset_type',type=str,choices=['classification', 'regression'],default='classification',
                   help='The type of dataset.')
    p.add_argument('--task_num',type=int,default=1,
                   help='The number of task in multi-task training.')
    p.add_argument('--split_type',type=str,choices=['random', 'scaffold'],default='random',
                   help='The type of data splitting.')
    p.add_argument('--split_ratio',type=float,nargs=3,default=[0.8,0.1,0.1],
                   help='The ratio of data splitting.[train,valid,test]')
    p.add_argument('--seed',type=int,default=30,
                   help='The random seed of model. Using in splitting data.')
    p.add_argument('--epochs',type=int,default=200,
                   help='The number of epochs.')
    p.add_argument('--batch',type=int,default=50,
                   help='The size of batch.')

def set_train_argument():
    p = ArgumentParser()
    add_train_argument(p)  #传终端输入的参数
    args = p.parse_args()
    
    # assert args.data_path
    # assert args.dataset_type
    
    # mkdir(args.save_path)
    
    # if args.metric is None:
    #     if args.dataset_type == 'classification': #这里定义了评估函数
    #         args.metric = 'auc'
    #     elif args.dataset_type == 'regression':
    #         args.metric = 'rmse'

    # if args.dataset_type == 'classification' and args.metric not in ['auc', 'prc-auc']:
    #     raise ValueError('Metric or data_type is error.')
    # if args.dataset_type == 'regression' and args.metric not in ['rmse']:
    #     raise ValueError('Metric or data_type is error.')
    # if args.fp_type not in ['mixed','morgan']:
    #     raise ValueError('Fingerprint type is error.')

    # args.cuda = torch.cuda.is_available()
    # args.init_lr = 1e-4
    # args.max_lr = 1e-3
    # args.final_lr = 1e-4
    # args.warmup_epochs = 2.0
    # args.num_lrs = 1
    
    return args





