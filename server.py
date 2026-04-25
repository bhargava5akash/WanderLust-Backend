from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from fastapi import FastAPI, APIRouter, Request, Response, HTTPException
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone, timedelta
from bson import ObjectId
import os, uuid, logging, bcrypt, jwt, json, secrets, httpx
from openai import OpenAI
import uvicorn
# ──────────────────── Config ────────────────────
ROOT_DIR = Path(__file__).parent
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

JWT_SECRET = os.environ['JWT_SECRET']
JWT_ALGORITHM = "HS256"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
api_router = APIRouter(prefix="/api")

# ──────────────────── Pydantic Models ────────────────────
class UserCreate(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class TripGenerateRequest(BaseModel):
    destination: str
    budget: Optional[float] = 30000
    duration: int = 5
    travel_style: str = "balanced"
    num_travelers: int = 2
    interests: List[str] = []
    travel_mode: str = "any"

class TripSaveRequest(BaseModel):
    trip_data: dict
    destination: str
    duration: int
    budget: Optional[float] = None

class BudgetRequest(BaseModel):
    destination: str
    duration: int
    num_travelers: int = 2
    travel_style: str = "balanced"
    travel_mode: str = "any"
    currency: str = "INR"

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# ──────────────────── Auth Utils ────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(user_id: str, email: str) -> str:
    payload = {"sub": user_id, "email": email, "exp": datetime.now(timezone.utc) + timedelta(minutes=60), "type": "access"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=7), "type": "refresh"}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        user["_id"] = str(user["_id"])
        user.pop("password_hash", None)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def set_auth_cookies(response: Response, access_token: str, refresh_token: str):
    response.set_cookie(key="access_token", value=access_token, httponly=True, secure=False, samesite="lax", max_age=3600, path="/")
    response.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=False, samesite="lax", max_age=604800, path="/")

# ──────────────────── Auth Routes ────────────────────
@api_router.post("/auth/register")
async def register(data: UserCreate, response: Response):
    email = data.email.lower().strip()
    existing = await db.users.find_one({"email": email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_doc = {
        "name": data.name,
        "email": email,
        "password_hash": hash_password(data.password),
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "favorites": [],
    }
    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)
    access_token = create_access_token(user_id, email)
    refresh_token = create_refresh_token(user_id)
    set_auth_cookies(response, access_token, refresh_token)
    return {"id": user_id, "name": data.name, "email": email, "role": "user"}

@api_router.post("/auth/login")
async def login(data: UserLogin, request: Request, response: Response):
    email = data.email.lower().strip()
    user = await db.users.find_one({"email": email})
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user_id = str(user["_id"])
    access_token = create_access_token(user_id, email)
    refresh_token = create_refresh_token(user_id)
    set_auth_cookies(response, access_token, refresh_token)
    return {"id": user_id, "name": user["name"], "email": email, "role": user.get("role", "user")}

@api_router.post("/auth/logout")
async def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return {"message": "Logged out"}

@api_router.get("/auth/me")
async def get_me(request: Request):
    user = await get_current_user(request)
    return user

@api_router.post("/auth/refresh")
async def refresh_token(request: Request, response: Response):
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user = await db.users.find_one({"_id": ObjectId(payload["sub"])})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        new_access = create_access_token(str(user["_id"]), user["email"])
        response.set_cookie(key="access_token", value=new_access, httponly=True, secure=False, samesite="lax", max_age=3600, path="/")
        return {"message": "Token refreshed"}
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

# ──────────────────── AI Trip Planner ────────────────────
TRIP_SYSTEM_PROMPT = """You are an expert India travel planner AI. You specialize in planning trips across India — from the Himalayas to the beaches of Goa, from the deserts of Rajasthan to the backwaters of Kerala.
Generate detailed travel itineraries focusing on Indian destinations, local experiences, regional cuisines, cultural festivals, and authentic travel.
All cost estimates MUST be in Indian Rupees (INR/₹). Include Indian transport options like trains, buses, bikes, and domestic flights.
Return ONLY valid JSON (no markdown, no backticks) in this exact format:
{
  "trip_name": "string",
  "destination": "string",
  "duration": number,
  "summary": "2-3 sentence trip summary highlighting local Indian experiences",
  "days": [
    {
      "day_number": 1,
      "title": "Day title",
      "activities": [
        {"time": "09:00", "activity": "Activity name", "description": "Brief description with local context", "cost_estimate": 500}
      ]
    }
  ],
  "budget_breakdown": {"accommodation": 0, "food": 0, "transport": 0, "activities": 0, "total": 0},
  "recommended_attractions": [{"name": "string", "description": "string", "type": "string"}],
  "food_places": [{"name": "string", "cuisine": "string (regional Indian cuisine)", "price_range": "string in ₹", "description": "string"}],
  "travel_tips": ["string - include India-specific tips"],
  "suggested_activities": [{"name": "string", "description": "string", "cost": "string in ₹"}],
  "transport_options": [{"mode": "train/bus/flight/bike/car", "route": "string", "duration": "string", "cost": "string in ₹"}]
}"""

@api_router.post("/trips/generate")
async def generate_trip(data: dict):

    user_message = f"""
    Create a detailed travel itinerary for {data.get('destination')}
    for {data.get('duration')} days.
    Budget: {data.get('budget')}
    Interests: {data.get('interests')}
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        print(response)

        response_text = response.choices[0].message.content if response.choices else "No response"

        return {
    "success": True,
    "trip": {
        "itinerary": response_text,
        "budget_breakdown": {
            "total": data["budget"],
            "accommodation": int(data["budget"] * 0.4),
            "food": int(data["budget"] * 0.2),
            "transport": int(data["budget"] * 0.3),
            "activities": int(data["budget"] * 0.1)
        },
        "food_recommendations": [
            "Try local street food",
            "Visit popular cafes",
            "Taste regional dishes"
        ],
        "travel_tips": [
            "Carry cash for local markets",
            "Start early for sightseeing",
            "Keep emergency contacts handy"
        ]
    }
}

    except Exception as e:
        logger.error(f"Chat error: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )

@app.post("/api/trips/save")
async def save_trip(request: Request):

    data = await request.json()

    trip_data = {
        "destination": data.get("destination"),
        "duration": data.get("duration"),
        "budget": data.get("budget"),
        "trip_data": data.get("trip_data"),
    }

    await trips_collection.insert_one(trip_data)

    return {
        "success": True
    }
trips_collection = db["trips"]

@api_router.get("/trips")
async def get_trips(request: Request):
    user = await get_current_user(request)
    trips = await db.trips.find({"user_id": user["_id"]}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return trips

@api_router.delete("/trips/{trip_id}")
async def delete_trip(trip_id: str, request: Request):
    user = await get_current_user(request)
    result = await db.trips.delete_one({"id": trip_id, "user_id": user["_id"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Trip not found")
    return {"message": "Trip deleted"}

# ──────────────────── Destinations ────────────────────
DESTINATIONS = [
    {"id": "manali", "name": "Manali, Himachal Pradesh", "state": "Himachal Pradesh", "description": "Snow-capped peaks, apple orchards, and adventure sports in the heart of the Himalayas.", "image": "https://images.unsplash.com/photo-1681176323164-bd4eeb724b81?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA1NzB8MHwxfHNlYXJjaHwxfHxNYW5hbGklMjBIaW1hY2hhbCUyMFByYWRlc2glMjBtb3VudGFpbnMlMjB2YWxsZXklMjBJbmRpYSUyMGdyZWVufGVufDB8fHx8MTc3Njk0Nzk2MHww&ixlib=rb-4.1.0&q=85", "category": "hill_station", "rating": 4.7, "avg_cost": 2500, "best_season": "Oct-Feb (Snow), Mar-Jun (Pleasant)", "tags": ["adventure", "nature", "trekking", "road_trip"]},
    {"id": "kashmir", "name": "Srinagar, Kashmir", "state": "Jammu & Kashmir", "description": "Paradise on earth with serene Dal Lake, Mughal gardens, and breathtaking valley views.", "image": "https://images.unsplash.com/photo-1716099934086-d64a79d43297?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NTYxOTB8MHwxfHNlYXJjaHwxfHxLYXNobWlyJTIwdmFsbGV5JTIwc2NlbmljJTIwYmVhdXR5JTIwSW5kaWF8ZW58MHx8fHwxNzc2OTQ3OTAzfDA&ixlib=rb-4.1.0&q=85", "category": "trending", "rating": 4.9, "avg_cost": 3000, "best_season": "Mar-Oct (Summer), Dec-Feb (Snow)", "tags": ["nature", "culture", "luxury", "houseboat"]},
    {"id": "goa", "name": "Goa", "state": "Goa", "description": "Sun-kissed beaches, vibrant nightlife, Portuguese heritage, and legendary seafood.", "image": "https://images.pexels.com/photos/8037061/pexels-photo-8037061.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940", "category": "beach", "rating": 4.6, "avg_cost": 2000, "best_season": "Nov-Feb (Peak), Sep-Oct (Monsoon charm)", "tags": ["beach", "nightlife", "food", "budget"]},
    {"id": "kerala", "name": "Alleppey, Kerala", "state": "Kerala", "description": "Tranquil backwaters, lush tea plantations, Ayurvedic wellness, and God's Own Country.", "image": "https://images.unsplash.com/photo-1593417034675-3ed7eda1bee9?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA2OTV8MHwxfHNlYXJjaHwyfHxLZXJhbGElMjBiYWNrd2F0ZXJzJTIwaG91c2Vib2F0JTIwSW5kaWF8ZW58MHx8fHwxNzc2OTQ3OTE0fDA&ixlib=rb-4.1.0&q=85", "category": "trending", "rating": 4.8, "avg_cost": 2500, "best_season": "Sep-Mar", "tags": ["nature", "luxury", "food", "wellness"]},
    {"id": "rajasthan", "name": "Jaipur, Rajasthan", "state": "Rajasthan", "description": "Majestic forts, vibrant bazaars, desert safaris, and royal Rajasthani hospitality.", "image": "https://images.unsplash.com/photo-1673115955449-4e50a5e78c9c?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA1NDh8MHwxfHNlYXJjaHwyfHxSYWphc3RoYW4lMjBwYWxhY2UlMjBkZXNlcnQlMjBJbmRpYSUyMEphaXB1cnxlbnwwfHx8fDE3NzY5NDc5MjB8MA&ixlib=rb-4.1.0&q=85", "category": "heritage", "rating": 4.7, "avg_cost": 2000, "best_season": "Oct-Mar", "tags": ["heritage", "culture", "food", "road_trip"]},
    {"id": "ladakh", "name": "Leh, Ladakh", "state": "Ladakh", "description": "Otherworldly landscapes, ancient monasteries, and the ultimate bike trip destination.", "image": "https://images.unsplash.com/photo-1756201409420-00f93939f9d1?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA0MTJ8MHwxfHNlYXJjaHwxfHxMYWRha2glMjBtb3VudGFpbiUyMGxhbmRzY2FwZSUyMGNpbmVtYXRpYyUyMEluZGlhfGVufDB8fHx8MTc3Njk0Nzg5Nnww&ixlib=rb-4.1.0&q=85", "category": "adventure", "rating": 4.9, "avg_cost": 3500, "best_season": "Jun-Sep", "tags": ["adventure", "trekking", "bike_trip", "spiritual"]},
    {"id": "meghalaya", "name": "Shillong, Meghalaya", "state": "Meghalaya", "description": "Living root bridges, pristine waterfalls, and the abode of clouds in Northeast India.", "image": "https://images.pexels.com/photos/18158726/pexels-photo-18158726.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940", "category": "hidden_gem", "rating": 4.7, "avg_cost": 2000, "best_season": "Oct-May", "tags": ["nature", "adventure", "hidden_gem", "trekking"]},
    {"id": "varanasi", "name": "Varanasi, Uttar Pradesh", "state": "Uttar Pradesh", "description": "India's spiritual capital — ancient ghats, sacred rituals, and the mystical Ganga Aarti.", "image": "https://images.pexels.com/photos/17869831/pexels-photo-17869831.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940", "category": "spiritual", "rating": 4.6, "avg_cost": 1500, "best_season": "Oct-Mar", "tags": ["spiritual", "culture", "food", "heritage"]},
    {"id": "rishikesh", "name": "Rishikesh, Uttarakhand", "state": "Uttarakhand", "description": "Yoga capital of the world, thrilling river rafting, and Himalayan foothills serenity.", "image": "https://images.pexels.com/photos/11521661/pexels-photo-11521661.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940", "category": "adventure", "rating": 4.7, "avg_cost": 1800, "best_season": "Sep-Nov, Feb-May", "tags": ["adventure", "spiritual", "wellness", "trekking"]},
    {"id": "hampi", "name": "Hampi, Karnataka", "state": "Karnataka", "description": "UNESCO ruins of the Vijayanagara Empire — boulders, ancient temples, and timeless beauty.", "image": "https://images.pexels.com/photos/3931344/pexels-photo-3931344.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940", "category": "heritage", "rating": 4.6, "avg_cost": 1200, "best_season": "Oct-Feb", "tags": ["heritage", "culture", "budget", "backpacking"]},
    {"id": "andaman", "name": "Andaman Islands", "state": "Andaman & Nicobar", "description": "Crystal clear waters, pristine beaches, scuba diving, and tropical island paradise.", "image": "https://images.pexels.com/photos/36505898/pexels-photo-36505898.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940", "category": "beach", "rating": 4.8, "avg_cost": 3000, "best_season": "Oct-May", "tags": ["beach", "adventure", "nature", "luxury"]},
    {"id": "coorg", "name": "Coorg, Karnataka", "state": "Karnataka", "description": "Misty coffee plantations, lush Western Ghats, and the Scotland of India.", "image": "https://images.pexels.com/photos/33046721/pexels-photo-33046721.png?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940", "category": "hidden_gem", "rating": 4.5, "avg_cost": 1800, "best_season": "Oct-Mar, Monsoon (Jun-Sep)", "tags": ["nature", "weekend_getaway", "food", "wellness"]},
]

@api_router.get("/destinations")
async def get_destinations(category: Optional[str] = None):
    if category and category != "all":
        return [d for d in DESTINATIONS if d["category"] == category]
    return DESTINATIONS

@api_router.get("/destinations/{dest_id}")
async def get_destination(dest_id: str):
    for d in DESTINATIONS:
        if d["id"] == dest_id:
            return d
    raise HTTPException(status_code=404, detail="Destination not found")

# ──────────────────── Budget Planner ────────────────────
# Daily costs per person in INR
COST_MULTIPLIERS = {
    "backpacking": {"accommodation": 500, "food": 300, "transport": 200, "activities": 200},
    "budget": {"accommodation": 1000, "food": 500, "transport": 400, "activities": 400},
    "balanced": {"accommodation": 2500, "food": 800, "transport": 600, "activities": 800},
    "luxury": {"accommodation": 8000, "food": 2000, "transport": 1500, "activities": 2000},
}

DESTINATION_PRICE_INDEX = {
    "goa": 1.0, "manali": 0.9, "kashmir": 1.1, "ladakh": 1.3, "kerala": 1.0,
    "rajasthan": 0.8, "varanasi": 0.6, "rishikesh": 0.8, "hampi": 0.5,
    "meghalaya": 0.7, "andaman": 1.4, "coorg": 0.9, "shimla": 0.9,
    "darjeeling": 0.8, "udaipur": 0.9, "jaisalmer": 0.8, "munnar": 0.9,
    "default": 1.0
}

@api_router.post("/budget/calculate")
async def calculate_budget(data: BudgetRequest):
    style = COST_MULTIPLIERS.get(data.travel_style, COST_MULTIPLIERS["balanced"])
    dest_lower = data.destination.lower()
    price_idx = DESTINATION_PRICE_INDEX.get("default", 1.0)
    for key, val in DESTINATION_PRICE_INDEX.items():
        if key in dest_lower:
            price_idx = val
            break
    breakdown = {}
    for category, daily_cost in style.items():
        breakdown[category] = round(daily_cost * price_idx * data.duration * data.num_travelers, 2)
    breakdown["total"] = round(sum(breakdown.values()), 2)
    breakdown["per_person"] = round(breakdown["total"] / data.num_travelers, 2)
    breakdown["per_day"] = round(breakdown["total"] / data.duration, 2)
    return {"destination": data.destination, "duration": data.duration, "travelers": data.num_travelers, "style": data.travel_style, "breakdown": breakdown}

# ──────────────────── AI Chat ────────────────────
CHAT_SYSTEM_PROMPT = """You are WanderLust AI, a friendly and expert India travel assistant. You specialize in travel within India. Help users with:
- Trip planning across Indian destinations (hill stations, beaches, heritage sites, spiritual places, Northeast India)
- Regional food recommendations (street food, local thalis, regional specialties)
- India-specific travel tips: trains (IRCTC), buses (RedBus), bike trips, road trips
- Budget advice in Indian Rupees (₹/INR) with backpacker to luxury ranges
- Festival timing (Diwali, Holi, Durga Puja, Onam, Pushkar Mela, etc.)
- Monsoon travel suggestions, Himalayan travel guidance, seasonal recommendations
- Safety tips, cultural etiquette, and local customs
Be warm, enthusiastic, and knowledgeable about India. Keep responses under 200 words unless asked for detail. Always quote prices in ₹ INR."""

@app.post("/api/chat")
async def chat(data: dict):

    try:
        user_message = data.get("message", "")

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        response_text = response.choices[0].message.content

        return {
    "itinerary": response_text
}

    except Exception as e:
        logger.error(f"Chat error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ──────────────────── Weather (OpenWeatherMap Live) ────────────────────
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
_weather_cache: dict = {}  # {city_lower: {"data": ..., "expires": datetime}}
CACHE_TTL = timedelta(minutes=30)

# City mapping for Indian destinations
DEST_CITY_MAP = {
    "manali": "Manali,IN", "himachal": "Shimla,IN", "shimla": "Shimla,IN",
    "kashmir": "Srinagar,IN", "srinagar": "Srinagar,IN", "gulmarg": "Gulmarg,IN",
    "goa": "Panaji,IN", "panaji": "Panaji,IN",
    "kerala": "Kochi,IN", "alleppey": "Alappuzha,IN", "munnar": "Munnar,IN", "kochi": "Kochi,IN",
    "rajasthan": "Jaipur,IN", "jaipur": "Jaipur,IN", "udaipur": "Udaipur,IN", "jaisalmer": "Jaisalmer,IN",
    "ladakh": "Leh,IN", "leh": "Leh,IN",
    "meghalaya": "Shillong,IN", "shillong": "Shillong,IN", "cherrapunji": "Cherrapunji,IN",
    "varanasi": "Varanasi,IN",
    "rishikesh": "Rishikesh,IN",
    "hampi": "Hampi,IN",
    "andaman": "Port Blair,IN",
    "coorg": "Madikeri,IN",
    "delhi": "New Delhi,IN", "mumbai": "Mumbai,IN", "bangalore": "Bangalore,IN",
    "kolkata": "Kolkata,IN", "chennai": "Chennai,IN", "hyderabad": "Hyderabad,IN",
    "darjeeling": "Darjeeling,IN", "gangtok": "Gangtok,IN", "ooty": "Ooty,IN",
}

def _resolve_city(destination: str) -> str:
    dl = destination.lower().strip()
    for key, city in DEST_CITY_MAP.items():
        if key in dl:
            return city
    return f"{destination},IN"

def _generate_travel_insights(temp: float, humidity: int, wind: float, condition: str, rain_prob: float, dest: str) -> dict:
    insights = []
    alerts = []
    dl = dest.lower()
    # Temperature insights
    if temp > 38:
        alerts.append("Extreme heat warning — stay hydrated, avoid outdoor activities 12-4 PM")
        insights.append("Best to explore early morning (6-10 AM) or evening (4-7 PM)")
    elif temp > 32:
        insights.append("Hot conditions — carry water, wear light breathable clothing")
        insights.append("Best exploration hours: early morning or late afternoon")
    elif temp < 5:
        alerts.append("Near-freezing temperatures — heavy winter gear essential")
        insights.append("Layer up with thermals, fleece, and windproof jacket")
    elif temp < 15:
        insights.append("Cool weather — carry a warm jacket and layers")
    else:
        insights.append("Pleasant weather for sightseeing and outdoor activities")
    # Rain
    if rain_prob > 70:
        alerts.append("High chance of rain — carry waterproof gear and umbrella")
        if any(k in dl for k in ["trek", "ladakh", "himachal", "rishikesh"]):
            alerts.append("Trekking safety alert: trails may be slippery, check conditions before heading out")
    elif rain_prob > 40:
        insights.append("Moderate rain chance — pack a light rain jacket")
    # Wind
    if wind > 30:
        alerts.append("Strong winds — avoid exposed mountain passes and high-altitude spots")
    # Beach
    if any(k in dl for k in ["goa", "andaman", "beach", "kovalam"]):
        if rain_prob > 60:
            insights.append("Beach conditions: rough seas likely, swimming may be unsafe")
        else:
            insights.append("Good beach conditions — ideal for water activities")
    # Mountain
    if any(k in dl for k in ["ladakh", "leh", "manali", "shimla", "darjeeling", "gangtok"]):
        if temp < 0:
            alerts.append("Sub-zero mountain conditions — roads may be icy or blocked")
        insights.append("Mountain weather changes quickly — carry layers regardless of forecast")
    return {"insights": insights, "alerts": alerts}

def _generate_packing_list(temp: float, humidity: int, rain_prob: float, condition: str, dest: str) -> list:
    items = []
    dl = dest.lower()
    # Universal
    items.append({"item": "Sunscreen SPF 50+", "reason": "UV protection essential across India"})
    items.append({"item": "Reusable water bottle", "reason": "Stay hydrated — refill at filtered water stations"})
    # Temperature-based
    if temp < 10:
        items.extend([
            {"item": "Thermal innerwear", "reason": f"Temperatures around {temp:.0f}°C — thermals are a must"},
            {"item": "Down jacket", "reason": "Insulation for cold mountain weather"},
            {"item": "Warm gloves & beanie", "reason": "Extremities lose heat fastest"},
            {"item": "Woolen socks", "reason": "Keep feet warm during treks and walks"},
        ])
    elif temp < 20:
        items.extend([
            {"item": "Light fleece jacket", "reason": "Cool evenings and mornings"},
            {"item": "Full-sleeve layers", "reason": f"Comfortable at {temp:.0f}°C with layering"},
        ])
    else:
        items.extend([
            {"item": "Cotton breathable clothing", "reason": f"Stay cool in {temp:.0f}°C heat"},
            {"item": "Cap / wide-brim hat", "reason": "Sun protection for outdoor exploration"},
        ])
    # Rain
    if rain_prob > 30:
        items.extend([
            {"item": "Compact umbrella", "reason": f"{rain_prob:.0f}% rain probability"},
            {"item": "Waterproof rain jacket", "reason": "Quick-dry protection from showers"},
            {"item": "Waterproof bag cover", "reason": "Keep electronics and documents dry"},
        ])
    # Humidity
    if humidity > 75:
        items.append({"item": "Quick-dry clothing", "reason": f"High humidity ({humidity}%) — cotton stays damp"})
    # Trekking destinations
    if any(k in dl for k in ["ladakh", "manali", "rishikesh", "meghalaya", "trek"]):
        items.extend([
            {"item": "Sturdy trekking shoes", "reason": "Essential for mountain trails and uneven terrain"},
            {"item": "First aid kit + Diamox", "reason": "Altitude sickness prevention for high-altitude areas"},
        ])
    # Beach destinations
    if any(k in dl for k in ["goa", "andaman", "beach"]):
        items.extend([
            {"item": "Swimwear", "reason": "Beach-ready essentials"},
            {"item": "Flip-flops / sandals", "reason": "Easy beach and town wear"},
        ])
    return items

async def _fetch_weather(city: str) -> dict:
    """Fetch current weather + 5-day forecast from OpenWeatherMap with caching."""
    cache_key = city.lower()
    now = datetime.now(timezone.utc)
    if cache_key in _weather_cache and _weather_cache[cache_key]["expires"] > now:
        return _weather_cache[cache_key]["data"]

    result = {}
    try:
        async with httpx.AsyncClient(timeout=10) as http:
            # Current weather
            curr_resp = await http.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
            )
            curr = curr_resp.json()
            if curr_resp.status_code != 200:
                raise Exception(curr.get("message", "API error"))

            # 5-day forecast
            fc_resp = await http.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"q": city, "appid": OPENWEATHER_KEY, "units": "metric"}
            )
            fc = fc_resp.json()

        # Parse current weather
        rain_vol = curr.get("rain", {}).get("1h", 0)
        clouds = curr.get("clouds", {}).get("all", 0)
        rain_prob = min(100, clouds * 0.8 + (50 if rain_vol > 0 else 0))
        result["current"] = {
            "temp_c": round(curr["main"]["temp"], 1),
            "feels_like": round(curr["main"]["feels_like"], 1),
            "condition": curr["weather"][0]["main"],
            "description": curr["weather"][0]["description"].title(),
            "icon": curr["weather"][0]["icon"],
            "humidity": curr["main"]["humidity"],
            "wind_kph": round(curr["wind"]["speed"] * 3.6, 1),
            "rain_probability": round(rain_prob),
            "pressure": curr["main"]["pressure"],
            "visibility": curr.get("visibility", 10000) // 1000,
        }

        # Parse 5-day forecast (one entry per day at noon)
        daily = {}
        for item in fc.get("list", []):
            dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            day_key = dt.strftime("%a")
            if day_key not in daily or "12:00" in item["dt_txt"]:
                pop = item.get("pop", 0) * 100
                daily[day_key] = {
                    "day": day_key,
                    "date": dt.strftime("%d %b"),
                    "high": round(item["main"]["temp_max"], 1),
                    "low": round(item["main"]["temp_min"], 1),
                    "condition": item["weather"][0]["main"],
                    "description": item["weather"][0]["description"].title(),
                    "icon": item["weather"][0]["icon"],
                    "rain_chance": round(pop),
                    "humidity": item["main"]["humidity"],
                    "wind_kph": round(item["wind"]["speed"] * 3.6, 1),
                }
        result["forecast"] = list(daily.values())[:5]
        result["source"] = "openweathermap"

        # Cache the result
        _weather_cache[cache_key] = {"data": result, "expires": now + CACHE_TTL}
    except Exception as e:
        logger.warning(f"OpenWeatherMap API failed for {city}: {e}")
        result = None
    return result

@api_router.get("/weather/{destination}")
async def get_weather(destination: str):
    city = _resolve_city(destination)
    data = await _fetch_weather(city)
    if not data:
        # Fallback — still generate rich insights from fallback data
        dl = destination.lower()
        fallback = WEATHER_FALLBACK.get("default")
        for key, val in WEATHER_FALLBACK.items():
            if key in dl:
                fallback = val
                break
        fb_current = fallback["current"]
        insights = _generate_travel_insights(fb_current["temp_c"], fb_current["humidity"], fb_current["wind_kph"], fb_current["condition"], fb_current["rain_probability"], destination)
        packing = _generate_packing_list(fb_current["temp_c"], fb_current["humidity"], fb_current["rain_probability"], fb_current["condition"], destination)
        return {"destination": destination, "source": "estimated", "current": fb_current,
                "forecast": fallback.get("forecast", []),
                "travel_insights": insights, "packing_suggestions": packing}

    current = data["current"]
    insights = _generate_travel_insights(current["temp_c"], current["humidity"], current["wind_kph"], current["condition"], current["rain_probability"], destination)
    packing = _generate_packing_list(current["temp_c"], current["humidity"], current["rain_probability"], current["condition"], destination)

    return {
        "destination": destination,
        "source": data["source"],
        "current": current,
        "forecast": data["forecast"],
        "travel_insights": insights,
        "packing_suggestions": packing,
    }

# Rich fallback data if API is down (still provides good UX)
WEATHER_FALLBACK = {
    "himachal": {"current": {"temp_c": 15, "feels_like": 13, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "humidity": 55, "wind_kph": 10, "rain_probability": 20, "pressure": 1015, "visibility": 10}, "forecast": [{"day": "Mon", "date": "", "high": 18, "low": 8, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 15, "humidity": 50, "wind_kph": 8}, {"day": "Tue", "date": "", "high": 16, "low": 7, "condition": "Clouds", "description": "Overcast", "icon": "04d", "rain_chance": 25, "humidity": 55, "wind_kph": 10}, {"day": "Wed", "date": "", "high": 19, "low": 9, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 45, "wind_kph": 7}, {"day": "Thu", "date": "", "high": 14, "low": 6, "condition": "Rain", "description": "Light Rain", "icon": "10d", "rain_chance": 60, "humidity": 70, "wind_kph": 12}, {"day": "Fri", "date": "", "high": 17, "low": 8, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 10, "humidity": 48, "wind_kph": 9}]},
    "kashmir": {"current": {"temp_c": 12, "feels_like": 10, "condition": "Clear", "description": "Clear Sky", "icon": "01d", "humidity": 50, "wind_kph": 8, "rain_probability": 10, "pressure": 1018, "visibility": 12}, "forecast": [{"day": "Mon", "date": "", "high": 15, "low": 5, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 45, "wind_kph": 6}, {"day": "Tue", "date": "", "high": 14, "low": 4, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 15, "humidity": 50, "wind_kph": 8}, {"day": "Wed", "date": "", "high": 16, "low": 6, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 42, "wind_kph": 7}, {"day": "Thu", "date": "", "high": 13, "low": 3, "condition": "Rain", "description": "Light Rain", "icon": "10d", "rain_chance": 50, "humidity": 65, "wind_kph": 10}, {"day": "Fri", "date": "", "high": 15, "low": 5, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 8, "humidity": 46, "wind_kph": 7}]},
    "goa": {"current": {"temp_c": 32, "feels_like": 35, "condition": "Clear", "description": "Sunny", "icon": "01d", "humidity": 70, "wind_kph": 12, "rain_probability": 10, "pressure": 1010, "visibility": 10}, "forecast": [{"day": "Mon", "date": "", "high": 33, "low": 26, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 68, "wind_kph": 10}, {"day": "Tue", "date": "", "high": 34, "low": 27, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 65, "wind_kph": 11}, {"day": "Wed", "date": "", "high": 32, "low": 25, "condition": "Thunderstorm", "description": "Thunderstorm", "icon": "11d", "rain_chance": 70, "humidity": 80, "wind_kph": 15}, {"day": "Thu", "date": "", "high": 33, "low": 26, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 10, "humidity": 70, "wind_kph": 12}, {"day": "Fri", "date": "", "high": 31, "low": 25, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 20, "humidity": 72, "wind_kph": 10}]},
    "kerala": {"current": {"temp_c": 28, "feels_like": 31, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "humidity": 80, "wind_kph": 8, "rain_probability": 40, "pressure": 1012, "visibility": 8}, "forecast": [{"day": "Mon", "date": "", "high": 30, "low": 24, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 15, "humidity": 75, "wind_kph": 7}, {"day": "Tue", "date": "", "high": 29, "low": 23, "condition": "Rain", "description": "Light Rain", "icon": "10d", "rain_chance": 55, "humidity": 85, "wind_kph": 9}, {"day": "Wed", "date": "", "high": 31, "low": 25, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 10, "humidity": 72, "wind_kph": 6}, {"day": "Thu", "date": "", "high": 28, "low": 23, "condition": "Thunderstorm", "description": "Thunderstorm", "icon": "11d", "rain_chance": 75, "humidity": 88, "wind_kph": 12}, {"day": "Fri", "date": "", "high": 30, "low": 24, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 12, "humidity": 74, "wind_kph": 7}]},
    "rajasthan": {"current": {"temp_c": 35, "feels_like": 37, "condition": "Clear", "description": "Sunny", "icon": "01d", "humidity": 25, "wind_kph": 15, "rain_probability": 5, "pressure": 1008, "visibility": 12}, "forecast": [{"day": "Mon", "date": "", "high": 38, "low": 25, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 2, "humidity": 22, "wind_kph": 14}, {"day": "Tue", "date": "", "high": 37, "low": 24, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 3, "humidity": 24, "wind_kph": 12}, {"day": "Wed", "date": "", "high": 39, "low": 26, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 2, "humidity": 20, "wind_kph": 16}, {"day": "Thu", "date": "", "high": 36, "low": 23, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 25, "wind_kph": 13}, {"day": "Fri", "date": "", "high": 38, "low": 25, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 3, "humidity": 23, "wind_kph": 14}]},
    "ladakh": {"current": {"temp_c": 8, "feels_like": 4, "condition": "Clear", "description": "Clear Sky", "icon": "01d", "humidity": 20, "wind_kph": 20, "rain_probability": 5, "pressure": 650, "visibility": 15}, "forecast": [{"day": "Mon", "date": "", "high": 12, "low": -2, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 2, "humidity": 18, "wind_kph": 18}, {"day": "Tue", "date": "", "high": 10, "low": -3, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 3, "humidity": 20, "wind_kph": 22}, {"day": "Wed", "date": "", "high": 13, "low": -1, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 2, "humidity": 16, "wind_kph": 17}, {"day": "Thu", "date": "", "high": 9, "low": -4, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 8, "humidity": 25, "wind_kph": 20}, {"day": "Fri", "date": "", "high": 11, "low": -2, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 3, "humidity": 19, "wind_kph": 19}]},
    "varanasi": {"current": {"temp_c": 30, "feels_like": 33, "condition": "Haze", "description": "Hazy", "icon": "50d", "humidity": 65, "wind_kph": 8, "rain_probability": 20, "pressure": 1010, "visibility": 5}, "forecast": [{"day": "Mon", "date": "", "high": 33, "low": 24, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 8, "humidity": 60, "wind_kph": 7}, {"day": "Tue", "date": "", "high": 32, "low": 23, "condition": "Haze", "description": "Hazy", "icon": "50d", "rain_chance": 12, "humidity": 65, "wind_kph": 6}, {"day": "Wed", "date": "", "high": 34, "low": 25, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 55, "wind_kph": 8}, {"day": "Thu", "date": "", "high": 31, "low": 22, "condition": "Rain", "description": "Light Rain", "icon": "10d", "rain_chance": 55, "humidity": 75, "wind_kph": 10}, {"day": "Fri", "date": "", "high": 33, "low": 24, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 10, "humidity": 58, "wind_kph": 7}]},
    "meghalaya": {"current": {"temp_c": 20, "feels_like": 20, "condition": "Clouds", "description": "Overcast", "icon": "04d", "humidity": 85, "wind_kph": 10, "rain_probability": 60, "pressure": 1014, "visibility": 6}, "forecast": [{"day": "Mon", "date": "", "high": 22, "low": 14, "condition": "Rain", "description": "Rain", "icon": "10d", "rain_chance": 70, "humidity": 88, "wind_kph": 9}, {"day": "Tue", "date": "", "high": 21, "low": 13, "condition": "Clouds", "description": "Cloudy", "icon": "04d", "rain_chance": 45, "humidity": 82, "wind_kph": 8}, {"day": "Wed", "date": "", "high": 23, "low": 15, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 15, "humidity": 70, "wind_kph": 7}, {"day": "Thu", "date": "", "high": 20, "low": 12, "condition": "Rain", "description": "Rain", "icon": "10d", "rain_chance": 75, "humidity": 90, "wind_kph": 11}, {"day": "Fri", "date": "", "high": 22, "low": 14, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 35, "humidity": 80, "wind_kph": 8}]},
    "rishikesh": {"current": {"temp_c": 26, "feels_like": 27, "condition": "Clear", "description": "Clear Sky", "icon": "01d", "humidity": 55, "wind_kph": 6, "rain_probability": 15, "pressure": 1014, "visibility": 10}, "forecast": [{"day": "Mon", "date": "", "high": 28, "low": 18, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 8, "humidity": 50, "wind_kph": 5}, {"day": "Tue", "date": "", "high": 27, "low": 17, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 20, "humidity": 55, "wind_kph": 7}, {"day": "Wed", "date": "", "high": 29, "low": 19, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 48, "wind_kph": 6}, {"day": "Thu", "date": "", "high": 25, "low": 16, "condition": "Rain", "description": "Light Rain", "icon": "10d", "rain_chance": 55, "humidity": 70, "wind_kph": 8}, {"day": "Fri", "date": "", "high": 28, "low": 18, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 10, "humidity": 52, "wind_kph": 5}]},
    "andaman": {"current": {"temp_c": 29, "feels_like": 32, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "humidity": 78, "wind_kph": 14, "rain_probability": 30, "pressure": 1011, "visibility": 10}, "forecast": [{"day": "Mon", "date": "", "high": 31, "low": 25, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 15, "humidity": 75, "wind_kph": 12}, {"day": "Tue", "date": "", "high": 30, "low": 24, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 25, "humidity": 78, "wind_kph": 13}, {"day": "Wed", "date": "", "high": 32, "low": 26, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 10, "humidity": 72, "wind_kph": 11}, {"day": "Thu", "date": "", "high": 29, "low": 24, "condition": "Rain", "description": "Rain", "icon": "10d", "rain_chance": 65, "humidity": 85, "wind_kph": 16}, {"day": "Fri", "date": "", "high": 30, "low": 25, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 20, "humidity": 76, "wind_kph": 13}]},
    "default": {"current": {"temp_c": 28, "feels_like": 30, "condition": "Clear", "description": "Sunny", "icon": "01d", "humidity": 60, "wind_kph": 10, "rain_probability": 15, "pressure": 1013, "visibility": 10}, "forecast": [{"day": "Mon", "date": "", "high": 30, "low": 22, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 8, "humidity": 55, "wind_kph": 8}, {"day": "Tue", "date": "", "high": 29, "low": 21, "condition": "Clouds", "description": "Partly Cloudy", "icon": "03d", "rain_chance": 20, "humidity": 60, "wind_kph": 9}, {"day": "Wed", "date": "", "high": 31, "low": 23, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 5, "humidity": 50, "wind_kph": 7}, {"day": "Thu", "date": "", "high": 28, "low": 20, "condition": "Rain", "description": "Light Rain", "icon": "10d", "rain_chance": 55, "humidity": 70, "wind_kph": 11}, {"day": "Fri", "date": "", "high": 30, "low": 22, "condition": "Clear", "description": "Sunny", "icon": "01d", "rain_chance": 10, "humidity": 58, "wind_kph": 8}]},
}

# ──────────────────── Currency Converter ────────────────────
@api_router.get("/currency/convert")
async def convert_currency(from_curr: str = "INR", to_curr: str = "USD", amount: float = 10000):
    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.get(f"https://api.frankfurter.app/latest?from={from_curr}&to={to_curr}&amount={amount}", timeout=10)
            data = resp.json()
            if "rates" in data:
                converted = data["rates"].get(to_curr, amount)
                return {"from": from_curr, "to": to_curr, "amount": amount, "converted": converted, "rate": round(converted / amount, 6)}
    except Exception as e:
        logger.error(f"Currency conversion error: {e}")
    # Fallback rates (from INR)
    fallback_rates = {"USD": 0.012, "EUR": 0.011, "GBP": 0.0095, "JPY": 1.79, "AUD": 0.018, "CAD": 0.016, "THB": 0.42, "IDR": 189.0, "SGD": 0.016, "MYR": 0.055, "LKR": 3.55, "NPR": 1.6, "BDT": 1.44}
    rate = fallback_rates.get(to_curr, 1.0)
    return {"from": from_curr, "to": to_curr, "amount": amount, "converted": round(amount * rate, 2), "rate": rate, "source": "fallback"}

# ──────────────────── Travel Insights ────────────────────
INSIGHTS = {
    "himachal": {"best_time": "Mar-Jun (Summer), Oct-Feb (Snow & Winter sports)", "safety": "Mountain roads can be tricky during monsoon. Carry warm clothing year-round at higher altitudes.", "culture": "Respectful of local Pahari culture. Himachali cap (topi) is a cultural symbol. Try Siddu and Dham.", "local_tips": "Book HRTC buses early. Carry cash — ATMs are scarce in remote areas. Rohtang Pass needs a permit.", "language": "Hindi, Pahari dialects. English understood in tourist areas."},
    "kashmir": {"best_time": "Mar-Oct for valleys, Dec-Feb for skiing in Gulmarg", "safety": "Generally safe for tourists. Follow local advisories. Register with local police if trekking.", "culture": "Kashmiri hospitality (Mehman Nawazi) is legendary. Try Wazwan feast. Visit Sufi shrines.", "local_tips": "Shikara ride on Dal Lake is a must. Bargain at floating markets. Book houseboats in advance.", "language": "Kashmiri, Urdu, Hindi. English in tourist areas."},
    "goa": {"best_time": "Nov-Feb (Peak beach season), Sep-Oct (Green monsoon charm)", "safety": "Safe for tourists. Avoid isolated beaches at night. Watch for strong currents during monsoon.", "culture": "Unique Indo-Portuguese culture. Visit old churches and spice plantations. Friday night markets are a must.", "local_tips": "Rent a scooter to explore. North Goa is lively, South Goa is serene. Try local Feni and fish curry rice.", "language": "Konkani, Hindi, English widely spoken."},
    "kerala": {"best_time": "Sep-Mar. Monsoon (Jun-Aug) for Ayurveda and lush green landscapes.", "safety": "Very safe. Watch for leeches during monsoon treks. Carry mosquito repellent.", "culture": "Rich temple culture. Kathakali performances are mesmerizing. Onam festival (Aug-Sep) is spectacular.", "local_tips": "Book Alleppey houseboat in advance. Visit Munnar tea gardens at sunrise. Try Kerala Sadhya (banana leaf meal).", "language": "Malayalam. English widely understood."},
    "rajasthan": {"best_time": "Oct-Mar (Winter). Avoid Apr-Jun extreme heat.", "safety": "Safe for tourists. Stay hydrated in summer. Be cautious of desert wildlife.", "culture": "Royal Rajput heritage. Colorful festivals (Pushkar Mela, Desert Festival). Try Dal Baati Churma and Laal Maas.", "local_tips": "Heritage hotels (Havelis) offer authentic stay. Take a camel safari in Jaisalmer. Visit forts early morning.", "language": "Rajasthani, Hindi. English in tourist areas."},
    "ladakh": {"best_time": "Jun-Sep (Roads open, clear skies). Winter is extreme.", "safety": "Acclimatize for 1-2 days in Leh (11,500 ft). AMS is real. Carry Diamox.", "culture": "Buddhist monasteries (Hemis, Thiksey, Diskit). Prayer flags everywhere. Respect local customs.", "local_tips": "Inner Line Permit needed for some areas. Fuel up before long rides. Carry cash — no ATMs beyond Leh.", "language": "Ladakhi, Hindi, English in Leh town."},
    "varanasi": {"best_time": "Oct-Mar. Dev Deepawali (Nov) is magical.", "safety": "Generally safe. Narrow lanes (galis) can be confusing — use local guides. Watch for scams at ghats.", "culture": "Oldest living city. Ganga Aarti at Dashashwamedh Ghat is unmissable. Silk weaving is a local art.", "local_tips": "Boat ride at dawn for best ghat views. Try street food at Kachori Gali. Visit Sarnath (Buddhist site) nearby.", "language": "Hindi, Bhojpuri. English understood at hotels."},
    "rishikesh": {"best_time": "Sep-Nov, Feb-May. Monsoon closes some treks.", "safety": "Safe. Follow safety instructions for rafting. Don't swim in strong Ganga currents.", "culture": "Yoga capital of the world. Beatles Ashram is iconic. Attend Triveni Ghat Aarti.", "local_tips": "Book rafting with certified operators. Laxman Jhula area for cafes. Try the 13-story Kunjapuri Temple sunrise trek.", "language": "Hindi, English (popular with international travelers)."},
    "meghalaya": {"best_time": "Oct-May. Monsoon is intense but beautiful.", "safety": "Safe and welcoming. Roads can be slippery in monsoon. Carry rain gear always.", "culture": "Matrilineal Khasi society. Living root bridges are unique to here. Try Jadoh (rice & pork) and Tungrymbai.", "local_tips": "Double Decker Root Bridge needs a 3000-step trek. Dawki river is clearest Oct-Nov. Hire local guides.", "language": "Khasi, Garo, Hindi, English widely spoken."},
    "default": {"best_time": "India is vast — each region has its ideal season. Generally Oct-Mar is best for most of India.", "safety": "India is generally safe for travelers. Use common sense, keep valuables secure, and stay aware.", "culture": "Incredibly diverse — every 100km brings new languages, food, and customs. Respect local traditions.", "local_tips": "Indian Railways is the best way to see the country. Book tickets on IRCTC. Try local street food.", "language": "Hindi and English are widely spoken. Each state has its own language."}
}

@api_router.get("/insights/{destination}")
async def get_insights(destination: str):
    dest_lower = destination.lower()
    insights = INSIGHTS.get("default")
    for key, val in INSIGHTS.items():
        if key in dest_lower:
            insights = val
            break
    return {"destination": destination, **insights}

# ──────────────────── Startup ────────────────────
@app.on_event("startup")
async def startup():
    # Create indexes
    await db.users.create_index("email", unique=True)
    await db.trips.create_index("user_id")
    await db.chat_messages.create_index("session_id")
    # Seed admin
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@wanderlust.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "WanderLust2024!")
    existing = await db.users.find_one({"email": admin_email})
    if not existing:
        await db.users.insert_one({
            "name": "Admin",
            "email": admin_email,
            "password_hash": hash_password(admin_password),
            "role": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "favorites": [],
        })
        logger.info(f"Admin user seeded: {admin_email}")
    elif not verify_password(admin_password, existing["password_hash"]):
        await db.users.update_one({"email": admin_email}, {"$set": {"password_hash": hash_password(admin_password)}})
        logger.info("Admin password updated")
    # Write test credentials
    (ROOT_DIR / "memory").mkdir(exist_ok=True)
    (ROOT_DIR / "memory" / "test_credentials.md").write_text(
        f"# Test Credentials\n\n## Admin\n- Email: {admin_email}\n- Password: {admin_password}\n- Role: admin\n\n## Auth Endpoints\n- POST /api/auth/register\n- POST /api/auth/login\n- POST /api/auth/logout\n- GET /api/auth/me\n- POST /api/auth/refresh\n"
    )
    logger.info("Startup complete")

# ──────────────────── Include Router & Middleware ────────────────────
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "https://wanderlust-frontend-9sl9.onrender.com"
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():

    import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    client.close()
