import firebase_admin
from firebase_admin import credentials, firestore
import bcrypt
from datetime import datetime

# Initialize Firebase (only once)
if not firebase_admin._apps:
    cred = credentials.Certificate('/app/firebase-key.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

def hash_password(password):
    """Hash a password for storing"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password, hashed):
    """Verify a stored password against one provided by user"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(username, password):
    """
    Register a new user
    
    Returns:
        dict: {'success': True, 'user_id': str} or {'success': False, 'error': str}
    """
    try:
        # Check if username already exists
        users_ref = db.collection('users')
        existing = users_ref.where('username', '==', username).limit(1).stream()
        
        if len(list(existing)) > 0:
            return {'success': False, 'error': 'Username already exists'}
        
        # Create new user document
        user_ref = users_ref.document()
        user_data = {
            'username': username,
            'password_hash': hash_password(password),
            'created_at': datetime.now(),
            'patterns': {}  # For time-of-day patterns
        }
        
        user_ref.set(user_data)
        
        print(f"✅ User registered: {username}")
        return {
            'success': True,
            'user_id': user_ref.id,
            'username': username
        }
        
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return {'success': False, 'error': str(e)}

def login_user(username, password):
    """
    Login a user
    
    Returns:
        dict: {'success': True, 'user_id': str, 'username': str} or {'success': False, 'error': str}
    """
    try:
        # Find user by username
        users_ref = db.collection('users')
        users = users_ref.where('username', '==', username).limit(1).stream()
        
        user_doc = None
        for user in users:
            user_doc = user
            break
        
        if not user_doc:
            return {'success': False, 'error': 'Invalid username or password'}
        
        user_data = user_doc.to_dict()
        
        # Verify password
        if not verify_password(password, user_data['password_hash']):
            return {'success': False, 'error': 'Invalid username or password'}
        
        print(f"✅ User logged in: {username}")
        return {
            'success': True,
            'user_id': user_doc.id,
            'username': username
        }
        
    except Exception as e:
        print(f"❌ Login error: {e}")
        return {'success': False, 'error': str(e)}

def get_user(user_id):
    """
    Get user data by user_id
    
    Returns:
        dict or None
    """
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            # Don't return password hash
            user_data.pop('password_hash', None)
            user_data['user_id'] = user_id
            return user_data
        return None
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return None