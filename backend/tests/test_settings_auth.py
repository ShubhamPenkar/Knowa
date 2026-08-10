"""Profile and workspace settings updates."""

from __future__ import annotations

import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.db.models import Organization, User
from app.services.auth_service import AuthService


class SettingsAuthFixture:
    def __init__(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db = self.Session()
        self.svc = AuthService(self.db)

        self.org = Organization(name="Settings Org", slug="settings-org", industry="saas")
        self.db.add(self.org)
        self.db.flush()

        self.owner = User(
            organization_id=self.org.id,
            email="owner@example.com",
            password_hash="x",
            name="Owner",
            role="owner",
        )
        self.member = User(
            organization_id=self.org.id,
            email="member@example.com",
            password_hash="x",
            name="Member",
            role="member",
        )
        self.db.add_all([self.owner, self.member])
        self.db.commit()

    def close(self):
        self.db.close()
        self.engine.dispose()


class TestSettingsAuth(unittest.TestCase):
    def setUp(self):
        self.fx = SettingsAuthFixture()

    def tearDown(self):
        self.fx.close()

    def test_update_user_profile(self):
        user = self.fx.svc.update_user_profile(self.fx.owner.id, name="  Ada Lovelace  ")
        self.assertEqual(user.name, "Ada Lovelace")

    def test_update_user_profile_rejects_blank(self):
        with self.assertRaises(ValueError):
            self.fx.svc.update_user_profile(self.fx.owner.id, name="   ")

    def test_update_organization(self):
        org = self.fx.svc.update_organization(
            self.fx.org.id,
            name="Knowa Labs",
            industry="finance",
        )
        self.assertEqual(org.name, "Knowa Labs")
        self.assertEqual(org.industry, "finance")

    def test_update_organization_rejects_bad_industry(self):
        with self.assertRaises(ValueError):
            self.fx.svc.update_organization(self.fx.org.id, industry="bananas")


if __name__ == "__main__":
    unittest.main()
