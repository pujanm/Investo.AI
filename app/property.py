import quandl

quandl.ApiConfig.api_key = "kyBMzL7tVaYQrYwmwW-m"
response = quandl.get('ZILL/Z90210_SPY')
print(response)