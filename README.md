# financial for python</br>
1、运行模拟数据</br>
打开文件目录输入命令如下：</br>
  python dqn_financial.py,即——使用默认参数训练</br>
  python dqn_financial.py --mode test ,就是测试模型</br>
2、训练真实数据</br>
打开文件目录输入命令如下：</br>
  python dqn_financial.py --env-name real --training-path EURUSD60_train.csv,即——使用默认参数训练,默认使用数据集EURUSD60_train.csv</br>
 python dqn_financial.py --mode test --env-name real --testing-path EURUSD60_test.csv 
 ,就是测试模型，默认使用数据集EURUSD60_test.csv</br>
#New Trader