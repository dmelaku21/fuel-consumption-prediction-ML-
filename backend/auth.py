import hashlib

# Simple user database (demo purpose)
USERS = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "Admin"
    },
    "operator": {
        "password": hashlib.sha256("operator123".encode()).hexdigest(),
        "role": "User"
    }
}

def authenticate(username, password):
    if username in USERS:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if USERS[username]["password"] == hashed:
            return True, USERS[username]["role"]
    return False, None
