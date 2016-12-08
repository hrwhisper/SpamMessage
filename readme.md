
## 环境 ##

- jieba
  - pip install jieba 
- sklearn
  - pip install -U scikit-learn
- python3.5



## 运行方法
- -c classfile_name 命令用来指定要用的分类器的名字：

| **名字**  | **对应的分类器**                  |
| ------- | --------------------------- |
| **p**   | Perceptron                  |
| **lr**  | LogisticRegression          |
| **nb**  | NaiveBayesian               |
| **svm** | SVM(sklearn)                |
| **lrs** | LogisticRegression(sklearn) |
| **nbs** | NaiveBayesian(sklearn)      |

- -i filename 为指定输入的短信文件名（该文件一行为一条短信）


- -o filename 为指定输出的结果文件（结果用0和1表示，1为垃圾短信，每一行对应输入文件的短信结果）

例如：

```
python judgeSpamMessage.py -c svm -i ./data/不带标签短信.txt -o ./data/result.txt
```

上述的命令指定了使用svm分类器，判断./data/不带标签短信.txt中的短信是否为垃圾短信，并将结果输出到./data/result.txt中。


## 训练啥的
- 首先运行token_and_save_to_file.py，分词保存结果
- test.py 中有交叉验证等方法



## 文件说明

文件夹解释如下：

| **文件夹名**       | **作用**      |
| -------------- | ----------- |
| **classifier** | 分类器代码存放的文件夹 |
| **data**       | 数据文件        |
| **model**      | 保存的模型       |

文件的解释如下：

| **文件夹名**                         | **作用**                                   |
| -------------------------------- | ---------------------------------------- |
| classifier/LogisticRegression.py | 本组实现的逻辑回归分类器源代码                          |
| classifier/NaiveBayesian.py      | 本组实现的朴素贝叶斯分类器源代码                         |
| classifier/Perceptron.py         | 本组实现的感知器分类器源代码                           |
| data/tags_token_results          | 带标签短信分词保存结果,token_and_save_to_file.py的生成的 |
| data/ tags_token_results_tag     | 带标签短信的类别                                 |
| data/不带标签短信.txt                  | 不带标签短信数据集                                |
| data/带标签短信.txt                   | 带标签短信数据集                                 |
| model/ Bayes_sklearn.pkl         | sklearn的贝叶斯分类器训练结果保存                     |
| model/ Logistic_sklearn.pkl      | sklearn的逻辑回归分类器训练结果保存                    |
| model/ LogisticRegression.pkl    | 本组实现的逻辑回归分类器训练结果保存                       |
| model/ NaiveBayesian.pkl         | 本组实现的贝叶斯分类器训练结果保存                        |
| model/ Perceptron.pkl            | 本组实现的感知器训练结果保存                           |
| model/ SVM_sklearn.pkl           | sklearn的SVM分类器结果保存                       |
| model/ train_data.pkl            | 带标签的短信的BOW表示结果                           |
| model/ vsm.pkl                   | 用于将新文档表示为BOW的训练完的类保存                     |
| judgeSpamMessage.py              | 用于判断输入的短信是否是垃圾短信                         |
| model_manage.py                  | 用于读入保存模型                                 |
| readme.md                        | 说明文件                                     |
| test.py                          | 测试文件                                     |
| token_and_save_to_file.py        | 分词并保存带标签的短信的结果，方便训练                      |