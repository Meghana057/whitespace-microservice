"""
Data Loader for HC.xlsx file - Clean Version for Modular Architecture
Processes real data and loads into PostgreSQL database
Trims to 4-5 organizations as per manager's requirements
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Import our database models
from database import SessionLocal, Account, Offering, initialize_database
from whitespace_ml import WhitespaceRecommendationEngine

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HCDataProcessor:
    def __init__(self, excel_file='HC.xlsx'):
        self.excel_file = excel_file
        self.selected_orgs = []
        self.processed_accounts = []
        self.all_offerings = []
        
    def load_excel_sheets(self):
        """Load all relevant sheets from HC.xlsx"""
        logger.info(f"Loading data from {self.excel_file}")
        
        if not Path(self.excel_file).exists():
            logger.error(f"File {self.excel_file} not found!")
            return None
        
        try:
            # Read the sheets we need
            excel_data = pd.read_excel(
                self.excel_file, 
                sheet_name=['Account Summary', 'All offerings'],
                engine='openpyxl'
            )
            
            account_data = excel_data['Account Summary']
            offerings_data = excel_data['All offerings']
            
            logger.info(f"Loaded {len(account_data)} accounts and {len(offerings_data)} offerings")
            
            return {
                'accounts': account_data,
                'offerings': offerings_data
            }
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            return None
    
    def create_organizations(self, accounts_df, max_orgs=5):
        """Create 4-5 organizations from the account data"""
        logger.info(f"Creating {max_orgs} organizations...")
        
        # Get unique industries, handling NaN values
        industries = accounts_df['Industry'].fillna('Technology').unique()
        
        # If we have fewer industries than desired orgs, create some defaults
        if len(industries) < max_orgs:
            default_industries = ['Technology', 'Healthcare', 'Manufacturing', 'Finance', 'Logistics']
            industries = list(industries) + [ind for ind in default_industries if ind not in industries]
            industries = industries[:max_orgs]
        else:
            # Take top industries by account count
            industry_counts = accounts_df['Industry'].fillna('Technology').value_counts()
            industries = industry_counts.head(max_orgs).index.tolist()
        
        # Create organization mapping
        org_templates = [
            ("Technology", "TechCorp Solutions"),
            ("Healthcare", "MedHealth Systems"), 
            ("Manufacturing", "IndustrialWorks Inc"),
            ("Finance", "FinanceFirst Corp"),
            ("Logistics", "LogiFlow Enterprises")
        ]
        
        self.selected_orgs = []
        for i, industry in enumerate(industries):
            # Use predefined names or generate one
            if i < len(org_templates):
                org_name = org_templates[i][1]
            else:
                org_name = f"{industry} Corporation"
            
            self.selected_orgs.append({
                'industry': industry,
                'organization_name': org_name
            })
        
        logger.info(f"Created organizations: {[org['organization_name'] for org in self.selected_orgs]}")
        return self.selected_orgs
    
    def process_accounts_data(self, accounts_df):
        """Process account data into our database format"""
        logger.info("Processing account data...")
        
        processed = []
        
        for idx, row in accounts_df.iterrows():
            try:
                # Basic validation
                if pd.isna(row.get('Name')) or str(row.get('Name')).strip() == '':
                    continue
                
                # Determine organization assignment
                account_industry = str(row.get('Industry', 'Technology')).strip()
                if account_industry == 'nan' or account_industry == '':
                    account_industry = 'Technology'
                
                # Find matching organization
                org_info = next(
                    (org for org in self.selected_orgs 
                     if org['industry'].lower() == account_industry.lower()),
                    self.selected_orgs[0] if self.selected_orgs else {
                        'organization_name': 'Default Corp', 
                        'industry': 'Technology'
                    }
                )
                
                # Process offerings
                offerings_sold = []
                if pd.notna(row.get('Offering sold')):
                    offerings_text = str(row['Offering sold']).strip()
                    if offerings_text and offerings_text not in ['', 'nan']:
                        # Clean and split offerings
                        offerings_sold = [
                            offering.strip() 
                            for offering in offerings_text.split(',')
                            if offering.strip() and offering.strip() != ''
                        ]
                
                # Calculate account age
                account_age_days = 365  # Default
                if pd.notna(row.get('CreatedDate')):
                    try:
                        if isinstance(row['CreatedDate'], str):
                            created_dt = pd.to_datetime(row['CreatedDate'])
                        else:
                            created_dt = row['CreatedDate']
                        
                        account_age_days = max(1, (datetime.now() - created_dt).days)
                    except:
                        account_age_days = 365
                
                # Helper function for safe numeric conversion
                def safe_numeric(val, default=0, is_float=False):
                    try:
                        if pd.isna(val) or val == '' or str(val).strip() == '':
                            return default
                        return float(val) if is_float else int(float(val))
                    except:
                        return default
                
                # Create processed account record
                account_record = {
                    'id': str(row.get('Id', f"acc_{idx:06d}")),
                    'name': str(row['Name']).strip(),
                    'industry': account_industry,
                    'organization_name': org_info['organization_name'],
                    'contacts': safe_numeric(row.get('Contacts')),
                    'active_opps': safe_numeric(row.get('Open Ops')),
                    'won_opps': safe_numeric(row.get('Won Ops')),
                    'lost_opps': safe_numeric(row.get('Lost Ops')),
                    'tasks': safe_numeric(row.get('Tasks')),
                    'events': safe_numeric(row.get('Events')),
                    'pipeline_revenue': safe_numeric(row.get('$ Open Ops'), is_float=True),
                    'won_revenue': safe_numeric(row.get('$ Won Ops'), is_float=True),
                    'lost_revenue': safe_numeric(row.get('$ Lost Ops'), is_float=True),
                    'is_existing_customer': True,  # All HC.xlsx accounts are existing
                    'account_age_days': account_age_days,
                    'offerings_sold': json.dumps(offerings_sold),
                    'created_date': datetime.now() - timedelta(days=account_age_days)
                }
                
                processed.append(account_record)
                
            except Exception as e:
                logger.warning(f"Error processing account {row.get('Name', 'Unknown')}: {str(e)}")
                continue
        
        self.processed_accounts = processed
        
        # Log organization distribution
        org_dist = {}
        for acc in processed:
            org = acc['organization_name']
            org_dist[org] = org_dist.get(org, 0) + 1
        
        logger.info(f"Processed {len(processed)} accounts")
        logger.info("Organization distribution:")
        for org, count in org_dist.items():
            logger.info(f"  {org}: {count} accounts")
        
        return processed
    
    def extract_all_offerings(self, accounts_df, offerings_df):
        """Extract all unique offerings from both data sources"""
        logger.info("Extracting all offerings...")
        
        offerings_set = set()
        
        # From dedicated offerings sheet
        if offerings_df is not None and 'Name' in offerings_df.columns:
            for _, row in offerings_df.iterrows():
                offering_name = row.get('Name')
                if pd.notna(offering_name) and str(offering_name).strip():
                    offerings_set.add(str(offering_name).strip())
        
        # From account data
        for _, row in accounts_df.iterrows():
            offerings_text = row.get('Offering sold')
            if pd.notna(offerings_text) and str(offerings_text).strip():
                offerings = [
                    o.strip() 
                    for o in str(offerings_text).split(',')
                    if o.strip()
                ]
                offerings_set.update(offerings)
        
        self.all_offerings = sorted([o for o in offerings_set if o and o != ''])
        logger.info(f"Found {len(self.all_offerings)} unique offerings")
        
        return self.all_offerings
    
    def load_to_database(self):
        """Load processed data into PostgreSQL database"""
        logger.info("Loading data to database...")
        
        db = SessionLocal()
        try:
            # Clear existing data
            logger.info("Clearing existing data...")
            db.query(Account).delete()
            db.query(Offering).delete()
            
            # Load offerings first
            logger.info(f"Loading {len(self.all_offerings)} offerings...")
            for offering_name in self.all_offerings:
                offering = Offering(
                    name=offering_name,
                    is_active=True
                )
                db.add(offering)
            
            # Load accounts
            logger.info(f"Loading {len(self.processed_accounts)} accounts...")
            for account_data in self.processed_accounts:
                account = Account(**account_data)
                db.add(account)
            
            db.commit()
            logger.info("Data loaded to database successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Database loading error: {str(e)}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def run_complete_process(self):
        """Run the complete data processing pipeline"""
        logger.info("="*60)
        logger.info("STARTING HC.XLSX DATA PROCESSING")
        logger.info("="*60)
        
        # Step 1: Load Excel data
        excel_data = self.load_excel_sheets()
        if excel_data is None:
            logger.error("Failed to load Excel data")
            return False
        
        # Step 2: Create organizations (4-5 as per manager requirement)
        self.create_organizations(excel_data['accounts'], max_orgs=5)
        
        # Step 3: Process accounts
        processed_accounts = self.process_accounts_data(excel_data['accounts'])
        if not processed_accounts:
            logger.error("No accounts processed")
            return False
        
        # Step 4: Extract offerings
        self.extract_all_offerings(excel_data['accounts'], excel_data['offerings'])
        
        # Step 5: Load to database
        if not self.load_to_database():
            logger.error("Database loading failed")
            return False
        
        # Success summary
        logger.info("="*60)
        logger.info("DATA PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Organizations created: {len(self.selected_orgs)}")
        logger.info(f"Accounts processed: {len(self.processed_accounts)}")
        logger.info(f"Offerings loaded: {len(self.all_offerings)}")
        logger.info("")
        logger.info("ORGANIZATIONS:")
        for org in self.selected_orgs:
            logger.info(f"  • {org['organization_name']} ({org['industry']})")
        
        return True

def initialize_and_train():
    """Initialize database and train models after data loading"""
    logger.info("Initializing ML models with loaded data...")
    
    try:
        # Initialize ML engine
        ml_engine = WhitespaceRecommendationEngine(threshold=0.5)
        
        # Get training data from database
        db = SessionLocal()
        accounts = db.query(Account).all()
        
        if len(accounts) < 5:
            logger.warning(f"Only {len(accounts)} accounts available for training")
            return False
        
        # Convert to training format
        training_data = []
        for acc in accounts:
            training_data.append({
                'id': acc.id,
                'name': acc.name,
                'industry': acc.industry,
                'organization_name': acc.organization_name,
                'contacts': acc.contacts,
                'active_opps': acc.active_opps,
                'won_opps': acc.won_opps,
                'lost_opps': acc.lost_opps,
                'tasks': acc.tasks,
                'events': acc.events,
                'pipeline_revenue': acc.pipeline_revenue,
                'won_revenue': acc.won_revenue,
                'lost_revenue': acc.lost_revenue,
                'is_existing_customer': acc.is_existing_customer,
                'account_age_days': acc.account_age_days,
                'offerings_sold': acc.offerings_sold
            })
        
        db.close()
        
        # Train models
        logger.info(f"Training models with {len(training_data)} accounts...")
        success = ml_engine.train_models(training_data)
        
        if success:
            # Save trained models
            from pathlib import Path
            Path("models").mkdir(exist_ok=True)
            ml_engine.save_models("models/whitespace_models.joblib")
            logger.info("Models trained and saved successfully!")
            return True
        else:
            logger.error("Model training failed")
            return False
            
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        return False

def main():
    """Main execution function"""
    logger.info("HC.xlsx Data Processor - FastAPI Microservice")
    logger.info("Processing real data for whitespace recommendations...")
    
    # Check if HC.xlsx exists
    if not Path('HC.xlsx').exists():
        logger.error("HC.xlsx file not found in current directory!")
        logger.info("Please place your HC.xlsx file in the same directory as this script")
        return False
    
    # Initialize database
    logger.info("Initializing database...")
    if not initialize_database():
        logger.error("Database initialization failed")
        return False
    
    # Process data
    processor = HCDataProcessor('HC.xlsx')
    if not processor.run_complete_process():
        logger.error("Data processing failed")
        return False
    
    # Train models
    if not initialize_and_train():
        logger.warning("Model training failed, but data is loaded. You can train later using the API.")
    
    # Final instructions
    print("\n" + "="*60)
    print("SETUP COMPLETE - READY TO START MICROSERVICE")
    print("="*60)
    print("Your HC.xlsx data has been processed and loaded.")
    print("Database is ready with accounts and offerings.")
    print("")
    print("NEXT STEPS:")
    print("1. Start the microservice: python main.py")
    print("2. Or with UV: uv run python main.py")
    print("3. API docs: http://localhost:8000/docs")
    print("4. Health check: http://localhost:8000/health")
    print("5. Test APIs: python test_api.py")
    print("")
    print("ORGANIZATIONS CREATED:")
    for org in processor.selected_orgs:
        accounts_in_org = len([a for a in processor.processed_accounts if a['organization_name'] == org['organization_name']])
        print(f"  • {org['organization_name']}: {accounts_in_org} accounts ({org['industry']} industry)")
    print("="*60)
    
    return True

if __name__ == "__main__":
    main()