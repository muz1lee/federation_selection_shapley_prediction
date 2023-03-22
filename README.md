
- 3/20 17:01 补充了流程中的utility function计算，分析计算相似度时需要。

## 联盟选择

分为训练和选择两部分

### 训练
本节会训练2个cifar10的联盟和2个cifar20的联盟。

#### cifar10

```shell
cd fedvar_inc/ # 处于fedvar_inc项目目录
# 2个cifar10的联盟：一个是dirichlet(\alpha=10)的数据划分，另一个是label(2)的数据划分
bash exps/exp1.7.1_pretrain_cifar10/run1a_train_old_federations_save_grad.sh # 这文件会输出所有要运行的python命令，并不会运行。
bash gstream.sh exps/exp1.7.1_pretrain_cifar10/run1a_train_old_federations_save_grad.sh # 这文件会运行相关的python命令。
# compute utility function
bash gstream.sh exps/exp1.7.1_pretrain_cifar10/run1b_compute_sv.sh
```

#### cifar20

```shell
cd fedvar_inc/ # 处于fedvar_inc项目目录
# 2个cifar20的联盟：一个是dirichlet(\alpha=10)的数据划分，另一个是label(2)的数据划分
bash exps/exp1.7.4_pretrain_cifar20/run1a_train_federation_save_grad.sh # 这文件会输出所有要运行的python命令，并不会运行，只是给你看看跑的是什么指令。
bash gstream.sh exps/exp1.7.4_pretrain_cifar20/run1a_train_federation_save_grad.sh # 这文件会运行相关的python命令。
# compute utility function
bash gstream.sh exps/exp1.7.4_pretrain_cifar20/run1b_compute_sv.sh
```

### 分析实验结果
在`fedvar_inc/exps/exp1.7.4_pretrain_cifar20/alg_select.ipynb`中，会对比4个联盟间的相似度。


## 其他说明
- `gstream.sh`是一个shell脚本，用于检测gpu是否有空位，自动为python命令分配gpu并运行。