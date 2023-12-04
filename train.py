from model import Network

from utils import load_data, split_data,smile_to_graph,MoleDataSet,set_train_argument,get_loss
import torch
import math
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 等变神经网络  非几何信息：点和边      几何信息：坐标
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')


def compute_mae_mse_rmse(target, prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    mae = sum(absError) / len(absError)  # 平均绝对误差MAE
    mse = sum(squaredError) / len(squaredError)  # 均方误差MSE
    RMSE = mse ** 0.5
    return mae, mse, RMSE


def compute_rsquared(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    r2 = round((SSR / SST) ** 2, 3)
    return r2

def process_data(data, iter_step):
    smile_used = 0
    automs_index_all = []
    feats_batch_all = []
    edges_batch_all = []
    coors_batch_all = []
    adj_batch_all = []
    mask_batch_all = []

    for i in range(0, len(data), iter_step):
        if smile_used + iter_step > len(data):  # 这个判断好像有点多余
            data_now = MoleDataSet(data[i:len(data)])
        else:
            data_now = MoleDataSet(data[i:i + iter_step])
        smile = data_now.smile()

        automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch = smile_to_graph(smile)
        automs_index_all.append(automs_index)
        feats_batch_all.append(feats_batch)
        edges_batch_all.append(edges_batch)
        coors_batch_all.append(coors_batch)
        adj_batch_all.append(adj_batch)
        mask_batch_all.append(mask_batch)

        smile_used += iter_step

    return {'automs_index_all': automs_index_all, 'feats_batch_all': feats_batch_all,
            'edges_batch_all': edges_batch_all,
            'coors_batch_all': coors_batch_all, 'adj_batch_all': adj_batch_all, 'mask_batch_all': mask_batch_all}

def predicting(model, val_data_dict, data, batch_size,dataset_type):
    model.eval()
    data_used = 0

    automs_index_all = val_data_dict.get('automs_index_all')
    feats_batch_all = val_data_dict.get('feats_batch_all')
    edges_batch_all = val_data_dict.get('edges_batch_all')
    coors_batch_all = val_data_dict.get('coors_batch_all')
    adj_batch_all = val_data_dict.get('adj_batch_all')
    mask_batch_all = val_data_dict.get('mask_batch_all')

    pred = []
    data_total = len(data)
    labels = data.label()
    target = torch.Tensor([[0 if x is None else x for x in tb] for tb in labels])

    count = 0
    for i in range(0, data_total, batch_size):

        data_now = MoleDataSet(data[i:i + batch_size])
        if len(data_now) < batch_size:
            batch_size = len(data_now)



        with torch.no_grad():
            #automs_index, feats_batch, edges_batch, coors_batch, adj_batch, mask_batch = smile_to_graph(smile)
            pred_now = model(automs_index_all[count], feats_batch_all[count], edges_batch_all[count],
                             coors_batch_all[count],
                             adj_batch_all[count], mask_batch_all[count], batch_size,dataset_type)

        count += 1
        data_used += batch_size
        pred_now = pred_now.data.cpu().numpy()

        pred_now = pred_now.tolist()
        pred.extend(pred_now)

    return np.array(pred).flatten(), np.array(target).flatten()


def train(model, data, train_data_dict, optimizer, loss_f, batch,dataset_type):
    model.train()

    iter_step = batch
    loss_sum = 0
    data_used = 0

    automs_index_all = train_data_dict.get('automs_index_all')
    feats_batch_all = train_data_dict.get('feats_batch_all')
    edges_batch_all = train_data_dict.get('edges_batch_all')
    coors_batch_all = train_data_dict.get('coors_batch_all')
    adj_batch_all = train_data_dict.get('adj_batch_all')
    mask_batch_all = train_data_dict.get('mask_batch_all')

    count = 0
    for i in range(0, len(data), iter_step):
        if data_used + iter_step > len(data):  # 这个判断好像有点多余
            data_now = MoleDataSet(data[i:len(data)])
            batch = len(data)-i
        else:
            data_now = MoleDataSet(data[i:i + iter_step])

        label = data_now.label()

        mask = torch.Tensor([[x is not None for x in tb] for tb in label])  # 这里为了防止有些无标签
        target = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])  # 这个target把无标签的当0处理

        if next(model.parameters()).is_cuda:
            mask, target = mask.cuda(), target.cuda()

        weight = torch.ones(target.shape).cuda()

        model.zero_grad()

        pred = model(automs_index_all[count], feats_batch_all[count], edges_batch_all[count], coors_batch_all[count],
                     adj_batch_all[count], mask_batch_all[count], batch,dataset_type)
        count = count + 1

        loss = loss_f(pred, target) * weight * mask
        loss = loss.sum() / mask.sum()
        loss_sum += loss.item()
        data_used += len(label)
        loss.backward()
        optimizer.step()


def epoch_train(args):
    # dataset_name='MDA-MB-231'
    # data_path = './data/BreastCellLines/'+dataset_name+'.csv'
    dataset_name = args.dataset_name
    data_path = args.data_dir + dataset_name
    # args = None
    data = load_data(data_path)
    split_ratio = args.split_ratio
    seed = args.seed
    batch_size = args.batch
    epochs=args.epochs
    split_type=args.split_type
    task_num=args.task_num
    dataset_type=args.dataset_type
    train_data, val_data, test_data = split_data(data, split_type, split_ratio, seed, None) #scaffold or random

    model = Network(task_num).cuda()

    encoder_file = str(epochs) + '_'+dataset_name+'_' + str(batch_size)
    model_load_path = args.save_dir+ encoder_file + '.pkl'
    if os.path.exists(model_load_path):
        model.load_state_dict(torch.load(model_load_path, map_location='cuda:0'))

    loss_fn = get_loss(dataset_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

    if dataset_type == 'classification':
        best_auc = 0
    else:
        best_auc = -10000
    stopping_monitor = 0
    independent_num = []
    valid_AUCs = args.log_dir + 'validAUCs_' + encoder_file + '.txt'

    # 训练数据集
    print('data process beginning')
    train_data_dict = process_data(train_data, batch_size)
    val_data_dict = process_data(val_data, batch_size)
    test_data_dict = process_data(test_data, batch_size)
    print('data process complete')
    for epoch in range(epochs):

        train(model, train_data, train_data_dict, optimizer, loss_fn, batch_size,dataset_type)

        if (epoch + 1) % 5 == 0:
            S, T = predicting(model, val_data_dict, val_data, batch_size,dataset_type)
            # T is correct score
            # S is predict score

            # compute preformence
            mae, mse, rmse = compute_mae_mse_rmse(T, S)
            r2 = compute_rsquared(S, T)

            R2 = r2_score(T, S)
            MSE = mean_squared_error(T, S)
            if dataset_type == 'classification':
                auc = roc_auc_score(T, S)
            else:
                auc = -1*MSE
            AUCs = [epoch, abs(auc), R2, mse, mae, rmse, r2]
            print(epoch)
            print('AUC: ', AUCs)

            if best_auc < auc:
                best_auc = auc
                stopping_monitor = 0
                print('best_auc：', best_auc)
                save_AUCs(AUCs, valid_AUCs)

                S, T = predicting(model,test_data_dict, test_data, batch_size,dataset_type)
                if dataset_type == 'classification':
                    auc = roc_auc_score(T, S)
                else:
                    auc = -1*mean_squared_error(T, S)
                print('test_AUC: ', auc)

                independent_num.append(T)
                independent_num.append(S)

                # 保存模型
                print('save model weights')
                torch.save(model.state_dict(), model_load_path)

            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 20:
                break

    # 最终效果最好的参数模型下载
    model.load_state_dict(torch.load(model_load_path, map_location='cuda:0'))
    S, T = predicting(model,test_data_dict, test_data, batch_size,dataset_type)
    mae, mse, rmse = compute_mae_mse_rmse(S, T)
    r2 = compute_rsquared(S, T)

    R2 = r2_score(T, S)
    MSE = mean_squared_error(T, S)
    if dataset_type == 'classification':
        auc = roc_auc_score(T, S)
    else:
        auc = -1*MSE
    AUCs = [0, abs(auc), R2, mse, mae, rmse, r2]
    print('test_AUC: ', AUCs)
    # save data
    #test_AUCs = '../result/log/'+dataset_name+'/testAUCs_' + encoder_file + '.txt'
    test_AUCs = args.log_dir + 'testAUCs_' + encoder_file + '.txt'
    save_AUCs(AUCs, test_AUCs)

if __name__ == '__main__':

    args = set_train_argument()
    print(args)
    # args.dataset_name='freesolv.csv'
    # args.data_dir='./data/Regression/'
    # args.dataset_type='regression'
    # python train.py --dataset_name 'freesolv.csv' --data_dir './data/Regression/' --dataset_type 'regression'







    epoch_train(args)





# feats_out, coors_out = net(feats_batch, coors_batch, edges = edges_batch, mask = mask_batch, adj_mat = adj_batch) # (1, 1024, 32), (1, 1024, 3)
# feats三维，coors三维，edges四维，mask三维

