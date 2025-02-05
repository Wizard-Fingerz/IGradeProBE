import redis

try:
    client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)
    client.ping()  # Should return True if Redis is reachable
    print("Redis connection successful!")
except redis.ConnectionError:
    print("Redis connection failed.")
