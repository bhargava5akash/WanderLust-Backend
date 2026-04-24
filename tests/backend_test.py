"""WanderLust India-first backend API tests - covers auth, destinations (Indian), INR budget, weather, insights, trips, AI."""
import os
import uuid
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://explore-smart-12.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"

ADMIN_EMAIL = "admin@wanderlust.com"
ADMIN_PASSWORD = "WanderLust2024!"


@pytest.fixture(scope="session")
def admin_session():
    s = requests.Session()
    r = s.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, timeout=20)
    assert r.status_code == 200, f"Admin login failed: {r.status_code} {r.text}"
    return s


@pytest.fixture(scope="session")
def new_user_session():
    s = requests.Session()
    email = f"TEST_{uuid.uuid4().hex[:8]}@wanderlust.test"
    r = s.post(f"{API}/auth/register", json={"name": "Test User", "email": email, "password": "TestPass123!"}, timeout=20)
    assert r.status_code == 200, f"Register failed: {r.status_code} {r.text}"
    s.email = email
    return s


# ───── Auth ─────
class TestAuth:
    def test_register_new_user(self):
        email = f"TEST_{uuid.uuid4().hex[:8]}@wanderlust.test"
        r = requests.post(f"{API}/auth/register", json={"name": "X", "email": email, "password": "Pass1234!"}, timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert data["email"] == email.lower()
        assert data["role"] == "user"

    def test_login_admin(self):
        s = requests.Session()
        r = s.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, timeout=20)
        assert r.status_code == 200
        assert r.json()["role"] == "admin"
        assert "access_token" in s.cookies

    def test_login_invalid(self):
        r = requests.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"}, timeout=20)
        assert r.status_code == 401

    def test_me_authenticated(self, admin_session):
        r = admin_session.get(f"{API}/auth/me", timeout=20)
        assert r.status_code == 200
        assert r.json()["email"] == ADMIN_EMAIL

    def test_me_unauthenticated(self):
        r = requests.get(f"{API}/auth/me", timeout=20)
        assert r.status_code == 401

    def test_logout(self):
        s = requests.Session()
        s.post(f"{API}/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, timeout=20)
        r = s.post(f"{API}/auth/logout", timeout=20)
        assert r.status_code == 200


# ───── Destinations (India-first) ─────
class TestDestinations:
    def test_list_all_12_indian(self):
        r = requests.get(f"{API}/destinations", timeout=20)
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d, list)
        assert len(d) == 12, f"Expected 12 Indian destinations, got {len(d)}"
        ids = {x["id"] for x in d}
        expected = {"manali", "kashmir", "goa", "kerala", "rajasthan", "ladakh",
                    "meghalaya", "varanasi", "rishikesh", "hampi", "andaman", "coorg"}
        assert expected.issubset(ids), f"Missing Indian dests: {expected - ids}"
        # INR pricing sanity - avg_cost should be in hundreds/thousands INR (not USD which'd be 10-200)
        for item in d:
            assert item["avg_cost"] >= 500, f"{item['id']} avg_cost {item['avg_cost']} looks non-INR"
            assert "state" in item

    def test_filter_hill_station(self):
        r = requests.get(f"{API}/destinations?category=hill_station", timeout=20)
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()]
        assert "manali" in ids

    def test_filter_beach(self):
        r = requests.get(f"{API}/destinations?category=beach", timeout=20)
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()]
        assert "goa" in ids
        assert "andaman" in ids

    def test_filter_spiritual(self):
        r = requests.get(f"{API}/destinations?category=spiritual", timeout=20)
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()]
        assert "varanasi" in ids

    def test_filter_heritage(self):
        r = requests.get(f"{API}/destinations?category=heritage", timeout=20)
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()]
        assert "rajasthan" in ids
        assert "hampi" in ids

    def test_filter_hidden_gem(self):
        r = requests.get(f"{API}/destinations?category=hidden_gem", timeout=20)
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()]
        assert "meghalaya" in ids
        assert "coorg" in ids

    def test_filter_adventure(self):
        r = requests.get(f"{API}/destinations?category=adventure", timeout=20)
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()]
        assert "ladakh" in ids or "rishikesh" in ids

    def test_get_by_id(self):
        r = requests.get(f"{API}/destinations/manali", timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "manali"
        assert "Himachal" in data["state"]

    def test_get_missing(self):
        r = requests.get(f"{API}/destinations/kyoto", timeout=20)
        assert r.status_code == 404


# ───── Budget (INR) ─────
class TestBudget:
    def test_calculate_goa_inr(self):
        r = requests.post(f"{API}/budget/calculate", json={
            "destination": "Goa", "duration": 5, "num_travelers": 2, "travel_style": "balanced", "currency": "INR"
        }, timeout=20)
        assert r.status_code == 200
        data = r.json()
        bd = data["breakdown"]
        for key in ["accommodation", "food", "transport", "activities", "total", "per_person", "per_day"]:
            assert key in bd
        # balanced/5d/2pax with price_idx 1.0: total ~= (2500+800+600+800)*5*2 = 47000 INR
        assert bd["total"] > 10000, f"INR total too low: {bd['total']}"

    def test_calculate_backpacking(self):
        r = requests.post(f"{API}/budget/calculate", json={
            "destination": "Hampi", "duration": 3, "num_travelers": 1, "travel_style": "backpacking"
        }, timeout=20)
        assert r.status_code == 200
        bd = r.json()["breakdown"]
        # backpacking Hampi (0.5): per_day ~= 600 INR
        assert bd["per_day"] < 2000


# ───── Currency ─────
class TestCurrency:
    def test_convert_inr_usd(self):
        r = requests.get(f"{API}/currency/convert?from_curr=INR&to_curr=USD&amount=10000", timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert data["from"] == "INR"
        assert data["to"] == "USD"
        assert data["converted"] > 0
        assert data["rate"] > 0


# ───── Weather (Indian, mocked) ─────
class TestWeather:
    def test_goa_monsoon(self):
        r = requests.get(f"{API}/weather/goa", timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "mocked"
        assert "temp_c" in data
        assert "monsoon_alert" in data
        assert "monsoon" in data["monsoon_alert"].lower() or "rain" in data["monsoon_alert"].lower()
        assert len(data["forecast"]) == 5

    def test_ladakh_cold(self):
        r = requests.get(f"{API}/weather/ladakh", timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert data["temp_c"] <= 15, f"Ladakh should be cold, got {data['temp_c']}"
        assert "monsoon_alert" in data


# ───── Insights (India) ─────
class TestInsights:
    def test_rajasthan_insights(self):
        r = requests.get(f"{API}/insights/rajasthan", timeout=20)
        assert r.status_code == 200
        data = r.json()
        assert "best_time" in data
        assert "safety" in data
        assert "culture" in data
        assert "language" in data
        # India-specific content check
        combined = (data["culture"] + data["local_tips"]).lower()
        assert any(t in combined for t in ["rajput", "rajasthan", "jaisalmer", "havelis", "dal baati"])


# ───── Trips ─────
class TestTrips:
    def test_save_and_get(self, admin_session):
        payload = {
            "trip_data": {"trip_name": "TEST India Trip", "destination": "Manali", "duration": 3},
            "destination": "Manali",
            "duration": 3,
            "budget": 30000,
        }
        r = admin_session.post(f"{API}/trips/save", json=payload, timeout=20)
        assert r.status_code == 200
        trip = r.json()
        assert trip["destination"] == "Manali"
        trip_id = trip["id"]

        r = admin_session.get(f"{API}/trips", timeout=20)
        assert r.status_code == 200
        assert any(t["id"] == trip_id for t in r.json())

        r = admin_session.delete(f"{API}/trips/{trip_id}", timeout=20)
        assert r.status_code == 200

    def test_trips_requires_auth(self):
        r = requests.get(f"{API}/trips", timeout=20)
        assert r.status_code == 401

    def test_generate_requires_auth(self):
        r = requests.post(f"{API}/trips/generate", json={"destination": "Manali", "duration": 3}, timeout=20)
        assert r.status_code == 401


# ───── AI ─────
class TestAI:
    def test_chat_india_focus(self, admin_session):
        r = admin_session.post(f"{API}/ai/chat", json={"message": "One short sentence: best month to visit Goa?"}, timeout=90)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "response" in data and len(data["response"]) > 0
