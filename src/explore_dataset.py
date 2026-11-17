from data_loader import download_dataset
import matplotlib.pyplot as plt
from data_analysis import analyze_correlation
import seaborn as sns
data=download_dataset()


print(data.head())
print("\nShape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nData types:")
print(data.dtypes)
print(data.isnull().sum())
print(data["MedHouseVal"].describe())
print(data.max())


# correlation=analyze_correlation(data)
# print("\n Correlation with house prices")
# print(correlation)
# correlation_matrix=data.corr(numeric_only=True)
# sns.heatmap(correlation_matrix,annot=True,fmt=".2f")
# plt.show()
# plt.scatter(
#     data["MedInc"],
#     data["MedHouseVal"],
#     alpha=0.3
# )

# plt.show()

from data_preparation import split_dataset
xtrain,xtest,ytrain,ytest=split_dataset(data,"MedHouseVal")
print("Training features:", xtrain.shape)
print("Testing features:", xtest.shape)

print("Training target:", ytrain.shape)
print("Testing target:", ytest.shape)

from model_training import build_model,train_model,build_random_forest,compare_train_test_performance,build_tuned_random_forest,perform_cross_validation

model=build_model()
model=train_model(model,xtrain,ytrain)
print("model training completed")

from model_training import evaluate_model
results=evaluate_model(model,xtest,ytest)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

rf_model=build_random_forest()
rf_model.fit(xtrain,ytrain)
results1=evaluate_model(rf_model,xtest,ytest)
print("\nRandom Forest results:")
for metric, value in results1.items():
    print(f"{metric}: {value:.4f}")



# comparison=compare_train_test_performance(rf_model,xtrain,ytrain,xtest,ytest)
# print("\nTrain vs Test Performance")
# for metric , value in comparison.items():
#     print(f"{metric}:{value:.4f}")


# rf_tuned_model=build_tuned_random_forest()
# rf_tuned_model.fit(xtrain,ytrain)
# results2=evaluate_model(rf_tuned_model,xtest,ytest)
# for metric,value in results.items():
#     print(f"{metric}: {value:.4f}")


# cv_scores=perform_cross_validation(rf_model,xtrain,ytrain)
# print("\nCross Validation Scores: ",cv_scores)
# print("Average CV R2:",cv_scores.mean())
# print("Standard Deviation: ",cv_scores.std())


# from model_training import tune_random_forest

# search = tune_random_forest(xtrain, ytrain)

# print("\nBest Parameters:")
# print(search.best_params_)

# print("\nBest Cross Validation R2:")
# print(search.best_score_)

# best_model = search.best_estimator_

# best_results = evaluate_model(
#     best_model,
#     xtest,
#     ytest
# )

# print("\nBest Model Test Results:")

# for metric, value in best_results.items():
#     print(f"{metric}: {value:.4f}")

# from experiment_tracking import log_experiment
# params=search.best_params_
# metrics = {
#     "MAE": best_results["MAE"],
#     "MSE": best_results["MSE"],
#     "RMSE": best_results["RMSE"],
#     "R2": best_results["R2 Score"],
#     "CV_R2": search.best_score_
# }

# # Original Random Forest
# log_experiment(
#     rf_model,
#     rf_model.get_params(),
#     results1,
#     "Original Random Forest"
# )

# # Manually tuned Random Forest
# log_experiment(
#     rf_tuned_model,
#     rf_tuned_model.get_params(),
#     results2,
#     "Manually Tuned Random Forest"
# )

# # RandomizedSearchCV model
# log_experiment(
#     best_model,
#     search.best_params_,
#     metrics,
#     "RandomizedSearchCV Random Forest"
# )



from model_saving import save_model,load_model
save_model(rf_model,"artifacts/house_price_rf.joblib")
loaded_model=load_model("artifacts/house_price_rf.joblib")
prediction=loaded_model.predict(xtest.iloc[[0]])
print("predictions : ",prediction[0])
print("actual value ",ytest.iloc[0])

