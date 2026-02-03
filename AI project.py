import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



#formula for accuracy
def accuracy(tpos,tneg,fpos,fneg):

    return((tpos+tneg)/(tpos+tneg+fpos+fneg))

#formula for sensitivity
def sensitivity(tpos,fneg):

    return(tpos/(tpos+fneg))

#formula for specificity
def specificity(tneg,fpos):

    return(tneg/(tneg+fpos))

#define all csv files
#update the paths to run locally
#cleve = pd.read_csv('\\Downloads\\heart+disease\\processed.cleveland.data', names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope' , 'ca', 'thal', 'num'] )
#hung = pd.read_csv("\\Downloads\\heart+disease\\processed.hungarian.data", names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope' , 'ca', 'thal', 'num'] )

#cleve = cleve.replace("?", np.NaN)
#hung = hung.replace("?", np.NaN)
#cleve.dropna(inplace=True)
#hung.dropna(inplace=True)

cleve = cleve.astype('float64', errors='ignore')
hung = hung.astype('float64', errors = 'ignore')

#combine the datasets into one data frame
frames = [cleve, hung]
data = pd.concat(frames)

#replace all ?'s with NaN values
data = data.replace('?', np.NaN)
data = data.astype('float64', errors = 'ignore')

#replace missing data using interpolation
data = data.interpolate(method='linear',limit_direction='forward')
data = round(data)

#sort data by gender
gendered_data = data.sort_values('sex')


#shuffle gendered data
final_data = gendered_data.sample(frac = 1, random_state=1)

#final_data ready to use

#uncomment below to see final_data
#print(final_data)
X = final_data.drop("num" , axis = 1)
#print(X)
Y = final_data['num']
#print(Y)


trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Scaling the data (Critical for Neural Networks)
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

model = mlp(hidden_layer_sizes=(300,), activation='relu', 
                      solver='adam', learning_rate = 'adaptive', max_iter=500, 
                      shuffle=False, verbose=False, warm_start=True, n_iter_no_change=15,
                      early_stopping=False)

model.fit(trainX, trainY)

trainY_pred = model.predict(trainX)
testY_pred = model.predict(testX)

print("Training Accuracy:", accuracy_score(trainY, trainY_pred, normalize=False))

TP = accuracy_score(testY, testY_pred, normalize=False)
print("True Positive: ", TP)
def confMatrixValues(actual, predict, num):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(predict)):
        if predict[i] == num and actual[i] == num:
            tp+=1
        if predict[i] == num and actual[i] != num:
            fp += 1
        if predict[i] != num and actual[i] != num:
            tn += 1
        if predict[i] != num and actual[i] == num:
            fn += 1

    return(tp,fp,tn,fn)


def plot_conf_matrix_values(actual, predict):
    num_classes = 5
    categories = list(range(num_classes))
    tp_values = []
    fp_values = []
    tn_values = []
    fn_values = []

    for i in categories:
        tp, fp, tn, fn = confMatrixValues(actual, predict, i)
        tp_values.append(tp)
        fp_values.append(fp)
        tn_values.append(tn)
        fn_values.append(fn)

    # Plot bar graph
    x = np.arange(len(categories))  # Category indices
    bar_width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * bar_width, tp_values, bar_width, label='TP', color='green')
    plt.bar(x - 0.5 * bar_width, fp_values, bar_width, label='FP', color='red')
    plt.bar(x + 0.5 * bar_width, tn_values, bar_width, label='TN', color='blue')
    plt.bar(x + 1.5 * bar_width, fn_values, bar_width, label='FN', color='orange')

    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('True Positive, False Positive, True Negative, and False Negative Counts')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


plot_conf_matrix_values(testY, testY_pred)


  


