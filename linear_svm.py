from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from util.utils import load_data,update_results_json
import argparse,os

# 解析命令行参数
parser = argparse.ArgumentParser(description='Decision Tree Classifier with PCA')
parser.add_argument('--pca_dim', type=int, default=50, help='Number of dimensions after PCA reduction')
args = parser.parse_args()

X_train, y_train, X_test, y_test = load_data('mnist_all.mat', args.pca_dim)
# 使用参数

# Initialize the Linear SVM model with a specified maximum number of iterations
max_iterations = 1000  # Adjust this value as needed
svm_model = LinearSVC(max_iter=max_iterations)
print('Initializing SVM model with max_iter =', max_iterations)

# Set flags for evaluation and saving the model
is_eval = False
is_save = False

# Load the MNIST dataset


svm_model.fit(X_train, y_train)
# Train the model
# Predict on the test set
y_pred = svm_model.predict(X_test)
print('Predicting on the test set')

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

os.makedirs('./experiments',exist_ok=True)
update_results_json(args.pca_dim,accuracy,'./experiments/svm.json')