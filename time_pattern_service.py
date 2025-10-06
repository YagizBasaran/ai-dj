from datetime import datetime
from auth_service import db

def get_time_section():
    """
    Get current time section based on hour of day
    
    Returns:
        str: time section name
    """
    hour = datetime.now().hour
    
    if 6 <= hour < 12:
        return 'morning_generation'
    elif 12 <= hour < 17:
        return 'in_job'
    elif 17 <= hour < 20:
        return 'after_job'
    elif 20 <= hour < 24:
        return 'evening_generation'
    else:  # 0 <= hour < 6
        return 'night_generation'

def track_song_play(user_id, song_data):
    """
    Track when a user plays a song at a specific time of day
    
    Args:
        user_id: str - Firebase user ID
        song_data: dict - must have 'mood' key
    
    Returns:
        bool: success or failure
    """
    if not user_id:
        return False  # Don't track for non-logged-in users
    
    try:
        time_section = get_time_section()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return False
        
        user_data = user_doc.to_dict()
        patterns = user_data.get('patterns', {})
        
        # Initialize time section if needed
        if time_section not in patterns:
            patterns[time_section] = {
                'moods': {},
                'total_plays': 0,
                'last_updated': datetime.now()
            }
        
        # Update mood count
        mood = song_data.get('mood', 'unknown')
        if mood not in patterns[time_section]['moods']:
            patterns[time_section]['moods'][mood] = 0
        
        patterns[time_section]['moods'][mood] += 1
        patterns[time_section]['total_plays'] += 1
        patterns[time_section]['last_updated'] = datetime.now()
        
        # Save back to Firestore
        user_ref.update({'patterns': patterns})
        
        print(f"✅ Tracked: {mood} in {time_section} for user {user_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error tracking song play: {e}")
        return False

def get_time_based_recommendations(user_id):
    """
    Get mood recommendations based on user's listening patterns for current time
    
    Args:
        user_id: str - Firebase user ID
    
    Returns:
        dict with recommended moods and metadata
    """
    if not user_id:
        return {
            'has_patterns': False,
            'time_section': get_time_section(),
            'recommended_moods': [],
            'message': 'Not logged in - no pattern tracking'
        }
    
    try:
        time_section = get_time_section()
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return {
                'has_patterns': False,
                'time_section': time_section,
                'recommended_moods': [],
                'message': 'User not found'
            }
        
        user_data = user_doc.to_dict()
        patterns = user_data.get('patterns', {})
        section_data = patterns.get(time_section)
        
        # Need at least 5 plays to make recommendations
        if not section_data or section_data.get('total_plays', 0) < 1:
            return {
                'has_patterns': False,
                'time_section': time_section,
                'recommended_moods': [],
                'total_plays': section_data.get('total_plays', 0) if section_data else 0,
                'message': f'Keep listening! Need 5+ plays in {time_section.replace("_", " ")} to learn your patterns.'
            }
        
        # Get top 2 moods for this time section
        moods = section_data['moods']
        sorted_moods = sorted(moods.items(), key=lambda x: x[1], reverse=True)
        top_moods = [mood for mood, count in sorted_moods[:2]]
        
        return {
            'has_patterns': True,
            'time_section': time_section,
            'recommended_moods': top_moods,
            'total_plays': section_data['total_plays'],
            'mood_breakdown': dict(sorted_moods[:3]),
            'message': f'Based on your {time_section.replace("_", " ")} habits'
        }
        
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
        return {
            'has_patterns': False,
            'time_section': get_time_section(),
            'recommended_moods': [],
            'message': 'Error loading patterns'
        }

def get_user_time_section_display():
    """
    Get a friendly display name for current time section
    
    Returns:
        str: formatted time section name
    """
    section = get_time_section()
    display_names = {
        'morning_generation': 'Morning (6am-12pm)',
        'in_job': 'Work Hours (12pm-5pm)',
        'after_job': 'After Work (5pm-8pm)',
        'evening_generation': 'Evening (8pm-12am)',
        'night_generation': 'Late Night (12am-6am)'
    }
    return display_names.get(section, section)