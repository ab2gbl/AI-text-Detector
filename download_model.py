import kagglehub
    
# Authenticate
kagglehub.login() # This will prompt you for your credentials.
# We also offer other ways to authenticate (credential file & env variables): https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate

path = kagglehub.model_download("abdessamiguebli/ai_detection_model_200k/pyTorch/default")

print("Path to model files:", path)
