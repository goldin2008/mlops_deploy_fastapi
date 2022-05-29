
# import joblib

# model_final = joblib.load("./model/trainedmodel.joblib")
# encoder = joblib.load("./model/encoder.joblib")
# lb = joblib.load("./model/lb.joblib")

# # Deploy model
# deploy_model(
#     model = model_final, 
#     model_name = 'model_test', 
#     platform = 'aws', 
#     authentication = {'bucket' : 'mlops-2022'}
# )
# # Enter your respective bucket name in place of 'mlopsdvc170100035'

# loaded_model = load_model(
#     'model_test', 
#     platform = 'aws', 
#     authentication = { 'bucket' : 'mlops-2022' }
# )

# data_unseen

# predictions = predict_model(loaded_model, data=data_unseen)
# predictions.head()    # View some of the predictions