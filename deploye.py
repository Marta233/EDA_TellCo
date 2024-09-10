import joblib

def save_model(model, filename='model.pkl'):
    joblib.dump(model, filename)

# Save the regression model
model, mse = build_regression_model()
save_model(model, 'regression_model.pkl')
