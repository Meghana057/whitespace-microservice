"""
Core Machine Learning System for Whitespace Identification
Separated from API layer for clean architecture
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
import joblib
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class WhitespaceRecommendationEngine:
    """
    Core ML engine for whitespace identification
    Based on your original whitespace.py but adapted for microservice architecture
    """
    
    def __init__(self, threshold: float = 0.5):
        # Model storage
        self.rf_models = {}  # Random Forest models for each product
        self.knn_model = None  # KNN model for similarity
        self.association_rules = {}  # Association rules
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Configuration
        self.threshold = threshold
        self.all_offerings = []
        self.feature_columns = [
            'contacts', 'active_opps', 'won_opps', 'lost_opps',
            'tasks', 'events', 'pipeline_revenue', 'won_revenue', 'lost_revenue',
            'is_existing_customer', 'account_age_days',
            'win_rate', 'total_opps', 'activity_score', 'engagement_ratio',
            'offerings_count', 'revenue_per_won_opp', 'pipeline_per_active_opp',
            'total_revenue_potential', 'account_age_years', 'customer_maturity_score',
            'industry_encoded'
        ]
        
        # Training status
        self.is_trained = False
        
    def create_features(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw account data
        Same logic as your original whitespace.py
        """
        logger.info("Creating engineered features...")
        
        feature_df = accounts_df.copy()
        
        # Handle division by zero with small epsilon
        epsilon = 0.001
        
        # Calculate derived metrics
        feature_df['win_rate'] = feature_df['won_opps'] / (
            feature_df['won_opps'] + feature_df['lost_opps'] + epsilon
        )
        feature_df['total_opps'] = (
            feature_df['active_opps'] + feature_df['won_opps'] + feature_df['lost_opps']
        )
        feature_df['activity_score'] = feature_df['tasks'] + feature_df['events']
        feature_df['engagement_ratio'] = feature_df['activity_score'] / (
            feature_df['contacts'] + epsilon
        )
        
        # Count current offerings
        def count_offerings(offerings_json):
            if pd.isna(offerings_json) or offerings_json == '':
                return 0
            try:
                offerings = json.loads(offerings_json)
                return len(offerings) if isinstance(offerings, list) else 0
            except:
                return 0
        
        feature_df['offerings_count'] = feature_df['offerings_sold'].apply(count_offerings)
        
        # Revenue metrics
        feature_df['revenue_per_won_opp'] = feature_df['won_revenue'] / (
            feature_df['won_opps'] + epsilon
        )
        feature_df['pipeline_per_active_opp'] = feature_df['pipeline_revenue'] / (
            feature_df['active_opps'] + epsilon
        )
        feature_df['total_revenue_potential'] = (
            feature_df['won_revenue'] + feature_df['pipeline_revenue']
        )
        
        # Customer maturity features
        feature_df['account_age_years'] = feature_df['account_age_days'] / 365.0
        feature_df['customer_maturity_score'] = (
            feature_df['account_age_years'] * 0.3 + 
            feature_df['win_rate'] * 0.4 + 
            feature_df['offerings_count'] * 0.3
        )
        
        # Encode categorical variables
        industries = feature_df['industry'].fillna('Unknown')
        if hasattr(self.label_encoder, 'classes_'):
            # Transform using existing encoder
            feature_df['industry_encoded'] = self.label_encoder.transform(industries)
        else:
            # Fit new encoder
            feature_df['industry_encoded'] = self.label_encoder.fit_transform(industries)
        
        logger.info(f"Created {len(self.feature_columns)} engineered features")
        return feature_df
    
    def extract_all_offerings(self, accounts_df: pd.DataFrame) -> List[str]:
        """Extract unique offerings from all accounts"""
        all_offerings_set = set()
        
        for _, row in accounts_df.iterrows():
            offerings_json = row.get('offerings_sold', '[]')
            if pd.notna(offerings_json) and offerings_json != '':
                try:
                    offerings = json.loads(offerings_json)
                    if isinstance(offerings, list):
                        all_offerings_set.update([str(o).strip() for o in offerings if str(o).strip()])
                except:
                    continue
        
        self.all_offerings = sorted(list(all_offerings_set))
        logger.info(f"Extracted {len(self.all_offerings)} unique offerings: {self.all_offerings}")
        return self.all_offerings
    
    def create_product_matrix(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        """Create binary matrix of accounts vs products"""
        logger.info("Creating product-account matrix...")
        
        product_matrix = accounts_df[['id', 'name']].copy()
        
        # Create binary columns for each offering
        for offering in self.all_offerings:
            product_matrix[f'has_{offering}'] = 0
        
        # Fill in the matrix
        for idx, row in accounts_df.iterrows():
            offerings_json = row.get('offerings_sold', '[]')
            if pd.notna(offerings_json) and offerings_json != '':
                try:
                    current_offerings = json.loads(offerings_json)
                    if isinstance(current_offerings, list):
                        for offering in current_offerings:
                            offering_clean = str(offering).strip()
                            if offering_clean in self.all_offerings:
                                col_name = f'has_{offering_clean}'
                                if col_name in product_matrix.columns:
                                    product_matrix.loc[idx, col_name] = 1
                except:
                    continue
        
        logger.info(f"Product matrix shape: {product_matrix.shape}")
        return product_matrix
    
    def train_random_forest_models(self, feature_matrix: np.ndarray, product_matrix: pd.DataFrame):
        """Train Random Forest model for each offering"""
        logger.info("Training Random Forest models...")
        
        successful_models = 0
        
        for offering in self.all_offerings:
            col_name = f'has_{offering}'
            if col_name not in product_matrix.columns:
                continue
            
            y = product_matrix[col_name].values
            positive_examples = y.sum()
            
            if positive_examples < 2:
                logger.warning(f"Skipping {offering} - only {positive_examples} positive examples")
                continue
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=3,
                random_state=42,
                class_weight='balanced'
            )
            
            try:
                rf.fit(feature_matrix, y)
                self.rf_models[offering] = rf
                successful_models += 1
                
                # Calculate cross-validation score if possible
                try:
                    cv_scores = cross_val_score(rf, feature_matrix, y, cv=min(3, positive_examples))
                    logger.info(f"✅ {offering}: CV Score = {cv_scores.mean():.3f}")
                except:
                    logger.info(f"✅ {offering}: Model trained")
                    
            except Exception as e:
                logger.error(f"Failed to train model for {offering}: {str(e)}")
                continue
        
        logger.info(f"Trained {successful_models} Random Forest models")
        return successful_models > 0
    
    def train_knn_model(self, feature_matrix: np.ndarray):
        """Train KNN model for similarity-based recommendations"""
        logger.info("Training KNN similarity model...")
        
        self.knn_model = NearestNeighbors(
            n_neighbors=min(10, len(feature_matrix)),
            metric='cosine',
            algorithm='brute'
        )
        
        self.knn_model.fit(feature_matrix)
        logger.info("✅ KNN model trained")
    
    def mine_association_rules(self, product_matrix: pd.DataFrame):
        """Simple association rule mining for product recommendations"""
        logger.info("Mining product association rules...")
        
        self.association_rules = {}
        
        for offering in self.all_offerings:
            has_col = f'has_{offering}'
            if has_col not in product_matrix.columns:
                continue
            
            # Find accounts that have this offering
            accounts_with_offering = product_matrix[product_matrix[has_col] == 1]
            
            if len(accounts_with_offering) == 0:
                continue
            
            # Find co-occurring offerings
            co_occurrence = {}
            for other_offering in self.all_offerings:
                if other_offering == offering:
                    continue
                
                other_col = f'has_{other_offering}'
                if other_col not in product_matrix.columns:
                    continue
                
                co_count = (accounts_with_offering[other_col] == 1).sum()
                if co_count > 0:
                    confidence = co_count / len(accounts_with_offering)
                    co_occurrence[other_offering] = confidence
            
            if co_occurrence:
                self.association_rules[offering] = co_occurrence
        
        logger.info(f"✅ Association rules mined for {len(self.association_rules)} offerings")
    
    def train_models(self, accounts_data: List[Dict]) -> bool:
        """
        Main training pipeline
        Takes list of account dictionaries from database
        """
        logger.info("Starting model training pipeline...")
        
        if not accounts_data:
            logger.error("No training data provided")
            return False
        
        # Convert to DataFrame
        accounts_df = pd.DataFrame(accounts_data)
        logger.info(f"Training on {len(accounts_df)} accounts")
        
        # Extract all unique offerings
        self.extract_all_offerings(accounts_df)
        
        if not self.all_offerings:
            logger.error("No offerings found in training data")
            return False
        
        # Create engineered features
        feature_df = self.create_features(accounts_df)
        
        # Scale numerical features
        feature_matrix = feature_df[self.feature_columns].values
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Create product matrix
        product_matrix = self.create_product_matrix(accounts_df)
        
        # Train models
        rf_success = self.train_random_forest_models(feature_matrix, product_matrix)
        if not rf_success:
            logger.error("Random Forest training failed")
            return False
        
        self.train_knn_model(feature_matrix)
        self.mine_association_rules(product_matrix)
        
        self.is_trained = True
        logger.info("✅ Model training completed successfully")
        return True
    
    def predict_for_account(self, account_data: Dict, top_n: int = 5) -> List[Dict]:
        """
        Generate whitespace recommendations for a single account
        Returns list of recommendations above threshold
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        logger.info(f"Generating recommendations for: {account_data.get('name')}")
        
        # Convert to DataFrame and create features
        account_df = pd.DataFrame([account_data])
        feature_df = self.create_features(account_df)
        
        # Scale features
        feature_matrix = self.scaler.transform(feature_df[self.feature_columns].values)
        
        # Get current offerings
        current_offerings = set()
        offerings_json = account_data.get('offerings_sold', '[]')
        if offerings_json:
            try:
                current_offerings = set(json.loads(offerings_json))
            except:
                current_offerings = set()
        
        recommendations = []
        
        # Random Forest predictions
        for offering, model in self.rf_models.items():
            if offering in current_offerings:
                continue  # Skip already owned products
            
            try:
                prob = model.predict_proba(feature_matrix)[0]
                confidence = prob[1] if len(prob) > 1 else prob[0]
                
                if confidence >= self.threshold:
                    recommendations.append({
                        'offering': offering,
                        'confidence_score': float(confidence),
                        'method': 'Random Forest',
                        'reasoning': f'ML model predicts {confidence:.3f} probability based on account characteristics',
                        'threshold_passed': True
                    })
            except Exception as e:
                logger.warning(f"RF prediction failed for {offering}: {str(e)}")
                continue
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} recommendations above threshold {self.threshold}")
        return recommendations[:top_n]
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        model_data = {
            'rf_models': self.rf_models,
            'knn_model': self.knn_model,
            'association_rules': self.association_rules,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'all_offerings': self.all_offerings,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.rf_models = model_data['rf_models']
            self.knn_model = model_data['knn_model']
            self.association_rules = model_data['association_rules']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.all_offerings = model_data['all_offerings']
            self.feature_columns = model_data['feature_columns']
            self.threshold = model_data['threshold']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about trained models"""
        return {
            'is_trained': self.is_trained,
            'num_rf_models': len(self.rf_models),
            'num_offerings': len(self.all_offerings),
            'threshold': self.threshold,
            'feature_count': len(self.feature_columns),
            'has_knn_model': self.knn_model is not None,
            'association_rules_count': len(self.association_rules)
        }