# Test Database Connection
# Run this script to test your database setup

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import app, db

def test_database():
    """Test database connection and create tables"""
    try:
        with app.app_context():
            print("Testing database connection...")
            db.create_all()
            print("✓ Database tables created successfully!")

            # Test basic operations
            from app import User
            test_user = User.query.first()
            print("✓ Database queries working!")

            print(f"Current database: {app.config['SQLALCHEMY_DATABASE_URI']}")
            print("✓ Database connection test passed!")

    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nTroubleshooting:")
        print("- For local development: Use DATABASE_URL=sqlite:///heart_data.db")
        print("- For Render deployment: Use the External Database URL from Render dashboard")
        print("- Make sure your Render database is active and accessible")

if __name__ == "__main__":
    test_database()