"""
Supabase Postgres ticket database — SQLAlchemy ORM model, session factory, and seed data.

Usage:
    python setup_db.py          # Creates table + seeds data (idempotent)

    from setup_db import Ticket, get_session
    session = get_session()
    tickets = session.query(Ticket).filter(Ticket.status == "open").all()
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import SUPABASE_DB_URL

# ---------------------------------------------------------------------------
# ORM Setup
# ---------------------------------------------------------------------------
Base = declarative_base()


class Ticket(Base):
    """Engineering ticket stored in Supabase Postgres."""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    status = Column(String(50))       # open | in_progress | resolved | closed
    assignee = Column(String(200))
    priority = Column(String(20))     # critical | high | medium | low
    component = Column(String(200))
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "assignee": self.assignee,
            "priority": self.priority,
            "component": self.component,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return f"<Ticket(id={self.id}, title='{self.title[:40]}...', status='{self.status}')>"


# ---------------------------------------------------------------------------
# Engine & Session Factory
# ---------------------------------------------------------------------------
engine = create_engine(
    SUPABASE_DB_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
SessionLocal = sessionmaker(bind=engine)


def get_session():
    """Return a new SQLAlchemy session. Caller must close it."""
    return SessionLocal()


# ---------------------------------------------------------------------------
# Seed Data
# ---------------------------------------------------------------------------
_SEED_TICKETS = [
    Ticket(
        title="Auth service returns 500 on token refresh",
        description="The auth-service /token/refresh endpoint intermittently returns HTTP 500. "
                    "Logs show a connection pool exhaustion to the user-sessions Redis cluster. "
                    "Affects ~2% of requests during peak hours.",
        status="open",
        assignee="alice@company.com",
        priority="critical",
        component="auth-service",
    ),
    Ticket(
        title="Upgrade Kubernetes cluster to 1.29",
        description="Scheduled upgrade of the production k8s cluster from 1.27 to 1.29. "
                    "Requires testing all CRDs and validating the Istio sidecar injection.",
        status="in_progress",
        assignee="bob@company.com",
        priority="high",
        component="infrastructure",
    ),
    Ticket(
        title="Migrate payment-service from Pydantic v1 to v2",
        description="Pydantic v2 has breaking changes in model_validator and Field defaults. "
                    "All schemas in payment-service/models/ need updating.",
        status="open",
        assignee="carol@company.com",
        priority="high",
        component="payment-service",
    ),
    Ticket(
        title="Add circuit breaker to order-service → inventory-service call",
        description="During the last incident (INC-2024-087), cascading failures between "
                    "order-service and inventory-service caused a 15-minute outage. "
                    "Implement Resilience4j circuit breaker pattern.",
        status="in_progress",
        assignee="dave@company.com",
        priority="high",
        component="order-service",
    ),
    Ticket(
        title="Flaky test: test_webhook_delivery_retry",
        description="test_webhook_delivery_retry in notification-service fails ~10% of the time "
                    "in CI due to race condition on the mock server port binding.",
        status="open",
        assignee="eve@company.com",
        priority="medium",
        component="notification-service",
    ),
    Ticket(
        title="Set up DataDog APM tracing for search-service",
        description="Instrument search-service with DataDog APM traces. Need to add dd-trace-py "
                    "and configure the Kubernetes deployment with the DD agent sidecar.",
        status="open",
        assignee="frank@company.com",
        priority="medium",
        component="search-service",
    ),
    Ticket(
        title="Database migration: add index on orders.customer_id",
        description="Query planner shows sequential scans on the orders table for customer lookup. "
                    "Adding a B-tree index on customer_id should reduce p99 latency from 800ms to <50ms.",
        status="resolved",
        assignee="grace@company.com",
        priority="high",
        component="database",
    ),
    Ticket(
        title="Update internal CLI tool (devctl) to support M1 Macs",
        description="devctl build scripts hardcode x86_64 architecture. Need cross-compilation "
                    "for arm64 and updated Homebrew formula.",
        status="resolved",
        assignee="heidi@company.com",
        priority="medium",
        component="developer-tools",
    ),
    Ticket(
        title="SSL certificate expiry alert for api.internal.company.com",
        description="The TLS certificate for api.internal.company.com expires in 14 days. "
                    "Need to rotate via cert-manager and verify all downstream services.",
        status="open",
        assignee="ivan@company.com",
        priority="critical",
        component="infrastructure",
    ),
    Ticket(
        title="Memory leak in recommendation-engine worker",
        description="recommendation-engine worker pods OOM-kill after ~6 hours. Heap dumps show "
                    "growing numpy arrays in the feature cache that are never released.",
        status="in_progress",
        assignee="judy@company.com",
        priority="critical",
        component="recommendation-engine",
    ),
    Ticket(
        title="Implement rate limiting on public API gateway",
        description="Need to add token-bucket rate limiting (100 req/min per API key) to the "
                    "Kong gateway. Configuration should be in the infrastructure-as-code repo.",
        status="open",
        assignee="karl@company.com",
        priority="high",
        component="api-gateway",
    ),
    Ticket(
        title="Onboarding doc: add section on VPN setup for remote engineers",
        description="New hires report confusion about the VPN setup process. Need to update "
                    "the onboarding guide in Confluence with step-by-step WireGuard instructions.",
        status="open",
        assignee="laura@company.com",
        priority="low",
        component="documentation",
    ),
    Ticket(
        title="CI pipeline timeout on integration tests",
        description="The integration test stage in the main pipeline times out at 30 minutes. "
                    "Root cause: the test database provisioning step takes 12 minutes due to "
                    "missing connection pooling in the test harness.",
        status="resolved",
        assignee="mallory@company.com",
        priority="medium",
        component="ci-cd",
    ),
    Ticket(
        title="Deprecate legacy /v1/users endpoint",
        description="The /v1/users REST endpoint is replaced by /v2/users (GraphQL). "
                    "Need to add deprecation headers, update API docs, and notify consumers "
                    "with a 90-day sunset timeline.",
        status="in_progress",
        assignee="nick@company.com",
        priority="medium",
        component="user-service",
    ),
    Ticket(
        title="Investigate Sentry spike: 'NoneType has no attribute get'",
        description="Sentry reports 4,200 occurrences of AttributeError in the last 24h "
                    "originating from notification-service/handlers/email.py:142. "
                    "Likely a null-check missing on the user preferences lookup.",
        status="open",
        assignee="olivia@company.com",
        priority="high",
        component="notification-service",
    ),
]


def setup_database() -> None:
    """Create the tickets table and seed it with sample data. Idempotent."""
    print("Creating tables …")
    Base.metadata.create_all(engine)

    session = get_session()
    try:
        existing = session.query(Ticket).count()
        if existing > 0:
            print(f"Tickets table already has {existing} rows — skipping seed.")
            return

        session.add_all(_SEED_TICKETS)
        session.commit()
        print(f"Seeded {len(_SEED_TICKETS)} tickets into Supabase.")
    except Exception as exc:
        session.rollback()
        print(f"Error seeding tickets: {exc}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    setup_database()
