import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import re

class SemanticMusicRecommender:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.scaler = StandardScaler()
        
        # Features for similarity calculation
        self.audio_features = [
            'energy', 'tempo', 'danceability', 'loudness', 'liveness', 
            'valence', 'speechiness', 'instrumentalness', 'acousticness'
        ]
        
        # Prepare the dataset
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        """Prepare embeddings and scaled features for the dataset"""
        
        # Create rich text descriptions for each track
        self.df['text_description'] = self.df.apply(self._create_track_description, axis=1)
        
        # Generate embeddings for all track descriptions
        print("Generating embeddings for tracks...")
        descriptions = self.df['text_description'].tolist()
        self.track_embeddings = self.model.encode(descriptions)
        
        # Scale audio features
        feature_data = self.df[self.audio_features].fillna(0)
        self.scaled_features = self.scaler.fit_transform(feature_data)
        
        print(f"Prepared {len(self.df)} tracks with embeddings and scaled features")
        
    def _create_track_description(self, row) -> str:
        """Create rich text description for each track to enable semantic matching"""
        
        # Basic info
        desc_parts = [f"Song by {row['track_artist']}"]
        
        # Mood/emotion
        if pd.notna(row['mood']) and row['mood'] != 'other':
            desc_parts.append(f"feels {row['mood']}")
            
        # Energy level
        energy = row['energy']
        if energy > 0.7:
            desc_parts.append("high energy, energetic, intense")
        elif energy > 0.4:
            desc_parts.append("moderate energy, balanced")
        else:
            desc_parts.append("low energy, calm, mellow")
            
        # Valence (positivity)
        valence = row['valence']
        if valence > 0.7:
            desc_parts.append("happy, positive, uplifting, joyful")
        elif valence > 0.3:
            desc_parts.append("neutral mood, balanced emotion")
        else:
            desc_parts.append("sad, melancholic, dark, negative, depressing")
            
        # Tempo
        tempo = row['tempo']
        if tempo > 140:
            desc_parts.append("fast tempo, quick, rapid, high-speed")
        elif tempo > 100:
            desc_parts.append("medium tempo, moderate pace")
        else:
            desc_parts.append("slow tempo, slow pace, relaxed timing")
            
        # Danceability
        if row['danceability'] > 0.7:
            desc_parts.append("danceable, groovy, rhythmic, party music")
        elif row['danceability'] < 0.3:
            desc_parts.append("not danceable, contemplative, listening music")
            
        # Acousticness
        if row['acousticness'] > 0.7:
            desc_parts.append("acoustic, organic, natural instruments")
        elif row['acousticness'] < 0.2:
            desc_parts.append("electronic, synthetic, produced")
            
        # Instrumentalness
        if row['instrumentalness'] > 0.5:
            desc_parts.append("instrumental, no vocals, no singing")
        else:
            desc_parts.append("has vocals, singing, lyrical")
            
        # Loudness
        if row['loudness'] > -5:
            desc_parts.append("loud, powerful, aggressive")
        elif row['loudness'] < -15:
            desc_parts.append("quiet, soft, gentle, subtle")
            
        # Speechiness
        if row['speechiness'] > 0.3:
            desc_parts.append("spoken word, rap, talking, speech-like")
        
        return ". ".join(desc_parts)
        
    def find_similar_tracks_semantic(self, prompt: str, top_k: int = 50) -> List[Dict]:
        """Find tracks semantically similar to the prompt"""
        
        # Generate embedding for the prompt
        prompt_embedding = self.model.encode([prompt])
        
        # Calculate semantic similarities
        similarities = cosine_similarity(prompt_embedding, self.track_embeddings)[0]
        
        # Get top similar tracks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            track = self.df.iloc[idx]
            results.append({
                'track_name': track['track_name'],
                'track_artist': track['track_artist'],
                'mood': track['mood'],
                'similarity_score': similarities[idx],
                'audio_features': {
                    'energy': track['energy'],
                    'valence': track['valence'],
                    'tempo': track['tempo'],
                    'danceability': track['danceability'],
                    'acousticness': track['acousticness']
                },
                'description': track['text_description']
            })
        
        return results
    
    def analyze_prompt_context(self, prompt: str) -> Dict:
        """Extract additional context from prompt without LLM"""
        
        text_lower = prompt.lower()
        analysis = {
            'detected_activities': [],
            'detected_emotions': [],
            'detected_preferences': [],
            'intensity_level': 'medium'
        }
        
        # Activity detection
        activity_keywords = {
            'studying': ['study', 'studying', 'homework', 'exam', 'focus', 'concentrate'],
            'working': ['work', 'working', 'office', 'coding', 'productive'],
            'exercising': ['gym', 'workout', 'running', 'exercise', 'fitness'],
            'relaxing': ['chill', 'relax', 'unwind', 'calm down', 'rest'],
            'partying': ['party', 'dancing', 'club', 'celebration'],
            'driving': ['driving', 'road trip', 'car', 'commute'],
            'sleeping': ['sleep', 'bedtime', 'night', 'tired']
        }
        
        for activity, keywords in activity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                analysis['detected_activities'].append(activity)
        
        # Emotion detection
        emotion_keywords = {
            'happy': ['happy', 'joyful', 'excited', 'cheerful', 'upbeat', 'positive'],
            'sad': ['sad', 'depressed', 'melancholy', 'down', 'blue', 'upset'],
            'angry': ['angry', 'furious', 'mad', 'rage', 'pissed', 'irritated'],
            'scared': ['scared', 'afraid', 'terrified', 'frightened', 'anxious', 'worried'],
            'energetic': ['energetic', 'pumped', 'hyped', 'motivated', 'intense'],
            'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'mellow'],
            'nostalgic': ['nostalgic', 'reminiscing', 'memories', 'throwback'],
            'romantic': ['romantic', 'love', 'intimate', 'date', 'crush']
        }
        
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                analysis['detected_emotions'].append(emotion)
        
        # Musical preferences
        preference_keywords = {
            'no_vocals': ['no vocals', 'instrumental', 'no singing'],
            'upbeat': ['upbeat', 'fast', 'energetic'],
            'slow': ['slow', 'mellow', 'gentle'],
            'loud': ['loud', 'heavy', 'aggressive'],
            'quiet': ['quiet', 'soft', 'ambient']
        }
        
        for pref, keywords in preference_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                analysis['detected_preferences'].append(pref)
        
        # Intensity detection
        intensity_words = ['very', 'extremely', 'really', 'totally', 'absolutely']
        exclamation_count = prompt.count('!')
        caps_ratio = sum(1 for c in prompt if c.isupper()) / max(len(prompt), 1)
        
        intensity_score = (
            sum(1 for word in intensity_words if word in text_lower) +
            exclamation_count * 0.5 +
            caps_ratio * 2
        )
        
        if intensity_score > 2:
            analysis['intensity_level'] = 'high'
        elif intensity_score < 0.5:
            analysis['intensity_level'] = 'low'
        
        return analysis
    
    def enhance_recommendations(self, semantic_results: List[Dict], 
                              prompt_analysis: Dict, top_k: int = 20) -> List[Dict]:
        """Enhance semantic results with contextual filtering and reranking"""
        
        enhanced_results = []
        
        for track in semantic_results:
            score = track['similarity_score']
            features = track['audio_features']
            
            # Context-based score adjustments
            context_bonus = 0
            
            # Activity-based adjustments
            if 'studying' in prompt_analysis['detected_activities']:
                # Prefer instrumental, low speechiness, moderate energy
                if features.get('acousticness', 0) > 0.5:
                    context_bonus += 0.1
                if track['mood'] in ['calm', 'other']:  # avoid distracting moods
                    context_bonus += 0.05
                if features.get('energy', 0) < 0.6:  # not too energetic
                    context_bonus += 0.05
                    
            elif 'exercising' in prompt_analysis['detected_activities']:
                # Prefer high energy, danceable
                if features.get('energy', 0) > 0.7:
                    context_bonus += 0.1
                if features.get('danceability', 0) > 0.6:
                    context_bonus += 0.05
                if features.get('tempo', 0) > 120:
                    context_bonus += 0.05
                    
            elif 'relaxing' in prompt_analysis['detected_activities']:
                # Prefer calm, acoustic, low energy
                if features.get('energy', 0) < 0.4:
                    context_bonus += 0.1
                if features.get('valence', 0) > 0.3:  # not too negative
                    context_bonus += 0.05
                if features.get('acousticness', 0) > 0.4:
                    context_bonus += 0.05
            
            # Emotion-based adjustments
            emotions = prompt_analysis['detected_emotions']
            if 'scared' in emotions or 'anxious' in emotions:
                # Prefer calming music
                if features.get('valence', 0) > 0.4 and features.get('energy', 0) < 0.5:
                    context_bonus += 0.1
                if track['mood'] == 'sad':  # might be too negative when scared
                    context_bonus -= 0.05
                    
            elif 'happy' in emotions:
                if track['mood'] == 'happy':
                    context_bonus += 0.1
                if features.get('valence', 0) > 0.6:
                    context_bonus += 0.05
                    
            elif 'sad' in emotions:
                if track['mood'] == 'sad':
                    context_bonus += 0.1
                if features.get('valence', 0) < 0.4:
                    context_bonus += 0.05
            
            # Preference-based adjustments
            prefs = prompt_analysis['detected_preferences']
            if 'no_vocals' in prefs:
                # This would need instrumentalness data, which seems missing in your sample
                # You could add logic here if you have that data
                pass
            
            # Calculate final score
            final_score = score + context_bonus
            
            track['final_score'] = final_score
            track['context_bonus'] = context_bonus
            enhanced_results.append(track)
        
        # Sort by final score and return top_k
        enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
        return enhanced_results[:top_k]
    
    def recommend(self, prompt: str, top_k: int = 20) -> Dict:
        """Main recommendation function"""
        
        # Get semantic similarities
        semantic_results = self.find_similar_tracks_semantic(prompt, top_k=100)
        
        # Analyze prompt context
        prompt_analysis = self.analyze_prompt_context(prompt)
        
        # Enhance with contextual information
        final_results = self.enhance_recommendations(
            semantic_results, prompt_analysis, top_k
        )
        
        return {
            'prompt': prompt,
            'recommendations': final_results,
            'analysis': prompt_analysis,
            'total_candidates': len(semantic_results)
        }

# Global recommender instance
_recommender = None

def get_recommender():
    """Get or initialize the recommender instance"""
    global _recommender
    if _recommender is None:
        _recommender = SemanticMusicRecommender('artifacts/v1/tracks_for_rec.csv')
    return _recommender

def recommend_from_prompt_semantic(prompt: str, topn: int = 20) -> Dict:
    """Bridge function for compatibility with ml_test2 route"""
    import time
    
    t0 = time.time()
    recommender = get_recommender()
    
    result = recommender.recommend(prompt, top_k=topn)
    
    # Format results to match expected structure
    formatted_results = []
    for track in result['recommendations']:
        formatted_results.append({
            'track_name': track['track_name'],
            'track_artist': track['track_artist']
        })
    
    return {
        'prompt': prompt,
        'results': formatted_results,
        'timing_ms': round((time.time() - t0) * 1000, 1),
        'analysis': result['analysis'],
        'total_candidates': result['total_candidates']
    }
