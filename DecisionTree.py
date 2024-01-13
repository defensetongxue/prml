from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from util.utils import load_data,update_results_json
import argparse,os

# 解析命令行参数
parser = argparse.ArgumentParser(description='Decision Tree Classifier with PCA')
parser.add_argument('--pca_dim', type=int, default=50, help='Number of dimensions after PCA reduction')
args = parser.parse_args()

# 使用参数
X_train, y_train, X_test, y_test = load_data('mnist_all.mat', args.pca_dim)

# Initialize the Decision Tree model with a maximum depth
dt_model = DecisionTreeClassifier(max_depth=10)

print('Initializing Decision Tree model')

# Train the model
dt_model.fit(X_train,y_train)

# Predict on the test set
y_pred = dt_model.predict(X_test)
print('Predicting on the test set')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

os.makedirs('./experiments',exist_ok=True)
update_results_json(args.pca_dim,accuracy,'./experiments/DecisionTree.json')