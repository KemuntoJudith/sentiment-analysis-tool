import time

start = time.time()
predict_sentiment("Test message")
end = time.time()

print(end - start)