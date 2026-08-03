#!/usr/bin/env python
"""Script to seed initial data into the database."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db, SessionLocal
from app.db.models import ActionCatalog, Customer, Organization, User
from app.services.auth_service import AuthService
from app.recommendations.action_catalog import ACTION_CATALOG
from app.ml.pipelines.preprocessing import generate_sample_data


def seed_action_catalog(db):
    """Seed action catalog."""
    print("Seeding action catalog...")
    
    existing = db.query(ActionCatalog).count()
    if existing > 0:
        print(f"  Action catalog already has {existing} entries, skipping")
        return
    
    for action in ACTION_CATALOG.values():
        catalog_entry = ActionCatalog(
            action_code=action.code,
            action_name=action.name,
            description=action.description,
            base_cost=action.base_cost,
            applicable_conditions=action.applicable_conditions,
        )
        db.add(catalog_entry)
    
    db.commit()
    print(f"  Added {len(ACTION_CATALOG)} actions")


def seed_sample_customers(db, n_customers: int = 100):
    """Seed sample customers."""
    print(f"Seeding {n_customers} sample customers...")
    
    existing = db.query(Customer).count()
    if existing > 0:
        print(f"  Already have {existing} customers, skipping")
        return
    
    # Generate sample data
    df = generate_sample_data(n_samples=n_customers)
    
    for i, row in df.iterrows():
        features = {
            col: row[col] if not hasattr(row[col], 'item') else row[col].item()
            for col in df.columns if col != 'churn'
        }
        
        customer = Customer(
            external_id=f"CUST{str(i+1).zfill(4)}",
            features=features,
        )
        db.add(customer)
    
    db.commit()
    print(f"  Added {n_customers} customers")


def seed_demo_user(db):
    """Seed a demo organization and user for local login."""
    demo_email = "demo@example.com"
    demo_password = "demo123"
    demo_org_name = "Demo Organization"
    demo_org_slug = "demo"

    auth_service = AuthService(db)

    org = db.query(Organization).filter(Organization.slug == demo_org_slug).first()
    if not org:
        org = auth_service.create_organization(
            name=demo_org_name,
            slug=demo_org_slug,
            industry="saas",
        )

    existing_user = db.query(User).filter(User.email == demo_email).first()
    if existing_user:
        existing_user.organization_id = org.id
        existing_user.password_hash = auth_service.hash_password(demo_password)
        existing_user.is_active = True
        existing_user.role = "owner"
        db.commit()
        print("Demo user updated: demo@example.com / demo123")
        return

    auth_service.create_user(
        org_id=org.id,
        email=demo_email,
        password=demo_password,
        name="Demo Admin",
        role="owner",
    )
    print("Demo user created: demo@example.com / demo123")


def main():
    print("Initializing database...")
    init_db()
    
    db = SessionLocal()
    try:
        seed_action_catalog(db)
        seed_sample_customers(db)
        seed_demo_user(db)
        print("\nSeeding complete!")
    finally:
        db.close()


if __name__ == "__main__":
    main()
