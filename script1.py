import kagglehub

kagglehub.login()

path = kagglehub.competition_download(
    'house-prices-advanced-regression-techniques'
)

print("Dataset downloaded to:", path)