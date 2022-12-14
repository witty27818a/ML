# Machine Learning
Machine Learning course at NYCU

## HW1
### 概念: 線性迴歸
***1102ML_HW01.pdf***: spec  
***HW1.txt***: 資料點  
***HW1_np版本.py***: 有numpy函數寫成的版本  
***HW1_純手刻版本.py***: 真的完全手刻所有矩陣運算函數的版本  

## HW2
### 概念: 樸素貝葉斯分類器、在線學習
***HW2_ML***: MNIST的訓練/測試資料的data/label的壓縮檔和解壓縮  
***110ML_HW02.pdf***: spec  
***Naive_Bayesian_Classifier.py***: 樸素貝葉斯分類器  
***Online_Learning.py***: 在線學習  
***Proof.jpg***: Beta共軛先驗分布的證明推導  
***beta.txt***: 在線學習的輸入資料  
***discrete說明.png***: 樸素貝葉斯分類器的discrete模式的算法圖解示意

## HW3
### 概念: 貝氏線性迴歸、在線學習
***110ML_HW03.pdf***: spec  
***Walford's online algorithm.png***: 在線更新期望值和變異數的Walford演算法說明  
***bayesian_linear_regression_modify.py***: 貝氏線性迴歸  
***random_generator.py***: 手刻從常態抽樣，以及從某個多項式抽樣的函數  
***sequential_estimator.py***: 使用Walford演算法在線更新期望值和變異數

## HW4
### 概念: 最大期望演算法、邏輯迴歸
***110ML_HW04.pdf***: spec  
***EM.py***: 最大期望演算法  
***logistic.py***: 邏輯迴歸  
***random_generator.py***: 手刻從常態抽樣，以及從某個多項式抽樣的函數  
剩下兩個是MNIST的訓練資料的data和label

## HW5
### 概念: 高斯過程、支持向量機
***data***: 所有輸入資料  
***2022_Spring_ML_HW5.pdf***: spec  
***Gaussian Process.py***: 高斯過程  
***Report.pdf***: 報告  
***SVM_Q1.py***: 支持向量機 -> 測試不同核函數的效果  
***SVM_Q2.py***: 支持向量機 -> 用有懲罰的C-SVC，然後使用網格搜索來調整參數  
***SVM_Q3.py***: 支持向量機 -> 混用線性核和RBF核

## HW6
### 概念: K-means演算法、譜分類法
***data***: 所有輸入資料  
***2022_Spring_ML_HW6.pdf***: spec  
***Report.pdf***: 報告  
***discussion_try.py***: 報告中的discussion的延伸實驗  
***kernel_Kmeans.py***: 使用核方法的Kmeans演算法  
***spectral_clustering.py***: 譜分群法  

## HW7
### 概念: 主成分分析、線性判別分析、t隨機鄰近嵌入法
***2022_Spring_ML_HW6.pdf***: spec  
***KernelEigenFaces.py***: 主成分分析(普通/使用核方法)、線性判別分析(普通/使用核方法)
***Report.pdf***: 報告  
***Yale_Face_Database.zip***: 臉部資料集  
***observation_and_discussion.py***: 報告中的discussion的延伸實驗  
***tsne.py***: t-SNE方法的sample code
***tsne_python.zip***: 包含上述t-SNE sample code以及輸入資料的壓縮檔
