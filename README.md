# **Environment**

the model was trainesd on  NVIDIA GPU 3080.

The most important python packages are:

- python == 3.8.10
- torch == 1.8.1
- rdkit ==2022.9.2
- scikit-learn == 1.1.3
- numpy == 1.21.2
- pandas==1.5.2

For using our model more conveniently, we provide the environment file *<environment.txt>* to install environment directly.

---

# **AEGNN-M Usage**

### **Train Model**

Use train.py

Args:

- --dataset_name:The name of input CSV file.E.g. 'bbbp.csv'
- --data_dir : The path of input CSV file. *E.g. './data/MoleculeNet/'*
- --dataset_type : The type of dataset. *E.g. classification  or  regression*
- --save_dir : The path to save output model. *E.g. './result/model/'*
- --log_dir : The path to record and save the result of training. *E.g.' ./result/log/'*
- --dataset_type:The type of dataset.E.g.' classification'
- --task_num:The number of task in multi-task training.E.g.1

E.g.

run easily

`python train.py     `

for classification task

`python train.py  --dataset_name 'bbbp.csv' --data_dir './data/MoleculeNet/' --save_dir './result/model/'  --log_dir './result/log/' --dataset_type 'classification' --task_num  1    `

for regression task

`python train.py  --dataset_name 'freesolv.csv' --data_dir './data/Regression/' --save_dir './result/model/'  --log_dir './result/log/' --dataset_type 'regression' --task_num  1 --epochs 300    `

---

# **Data**set

We provide the public benchmark datasets used in our study: *<data.rar>*

- MoleculeNet contains seven classification datasets.E.g.BBBP,BACE.
- BreastCellLines contains 14 classification datasets.E.g.BT-20,Bcap37.
- Regression contains three regression datasets.Eg.ESOL,FreeSolv.



The dataset file is a **CSV** file with a header line and label columns. E.g.

```
SMILES,BT-20
O(C(=O)C(=O)NCC(OC)=O)C,0
FC1=CNC(=O)NC1=O,0
```



---

# Results

The meaning of the parameters printed in the log is as follows:

for classification task

```
[epoch, auc, R2, mse, mae, rmse, r2]
```

for regression task

```
[epoch, mse, R2, mse, mae, rmse, r2]
```

