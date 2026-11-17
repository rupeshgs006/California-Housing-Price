from sklearn.model_selection import train_test_split

def split_dataset(data,target_column):
    x=data.drop(columns=[target_column])
    y=data[target_column]

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
    return xtrain,xtest,ytrain,ytest

