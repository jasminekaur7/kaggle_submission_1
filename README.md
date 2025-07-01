# 🌾 Yield Prediction using PyCaret — Kaggle Submission 1

---

## 👩‍💻 About Me

Hi! I'm **Jasmine Kaur**, a passionate data enthusiast and aspiring machine learning engineer. I enjoy solving real-world problems using data science and automation tools.  
This project was created as part of a **Kaggle Summer School Hackathon**, and showcases an end-to-end ML pipeline for predicting agricultural yield.

- 🔗 [Kaggle Profile](https://www.kaggle.com/jasminekaur876)  
- 💼 [LinkedIn](https://www.linkedin.com/in/jasmine-kaur-463a3a324/)  
- 💻 [GitHub](https://github.com/jasminekaur7)  

---

## 🧩 Problem Statement

Given a dataset of environmental and biological parameters, the goal is to accurately predict **crop yield**. The dataset includes information about:

- **Pollinators** like honeybee, osmia, andrena, bumbles  
- **Climatic variables** such as temperature ranges, raining days  
- **Plant traits** like fruit mass, fruit set, clone size, etc.

We are provided with:

- `train.csv`: Training dataset  
- `test.csv`: Test dataset  
- `sample_submission.csv`: Format for final prediction submission  

---
🏁 Goal of This Repo
📍 Track my Kaggle progress and experiments
📍 Share reusable and explainable code with the community
📍 Keep an open portfolio for recruiters and collaborators
📍 Keep learning, keep iterating!

## 🧠 Machine Learning Approach

We used the **PyCaret** library to automate the machine learning workflow. Here's what the pipeline included:

- ✅ **Feature Selection** using `classic` method  
- ✅ **Normalization** of feature space  
- ✅ **Power Transformation** to stabilize variance  
- ✅ **Automatic Model Comparison** using `compare_models()`  
- ✅ **Model Used**: `Gradient Boosting Regressor (gbr)`  
- ✅ **Prediction**: Performed on test set and saved as `Answer.csv`

### Key Code Snippet

```python
from pycaret.regression import *

setup(data=mydataset, target='yield',
      feature_selection=True,
      feature_selection_method='classic',
      normalize=True,
      transformation=True)

best_model = compare_models()
final_model = create_model("gbr")
predictions = predict_model(final_model, data=mynewdata)

output = pd.DataFrame({
    'id': predictions['id'],
    'yield': predictions['prediction_label']
})
output.to_csv("Answer.csv", index=False)


