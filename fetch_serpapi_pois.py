# fetch_serpapi_pois.py (fix)
import requests
import pandas as pd
from typing import Optional

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"

PRICE_SYMBOL_MAP = {
    "$":   (20000, 70000),
    "$$":  (80000, 200000),
    "$$$": (200000, 500000),
    "$$$$":(500000, 1000000),
}

def _symbol_to_range(symbol: Optional[str]):
    if not symbol:
        return None
    return PRICE_SYMBOL_MAP.get(symbol)

def _extract_results(payload: dict):
    """
    Trả về danh sách địa điểm từ nhiều khả năng khác nhau:
    - local_results (phổ biến)
    - places_results (thi thoảng SerpApi trả ra kiểu này)
    - Nếu đều trống -> []
    """
    if isinstance(payload.get("local_results"), list):
        return payload["local_results"]
    if isinstance(payload.get("places_results"), list):
        return payload["places_results"]
    return []

def get_serpapi_pois(
    lat: float,
    lng: float,
    keyword: str,
    api_key: str,
    radius_km: float = 3.0,  # không dùng trực tiếp bởi SerpApi; để giữ API tương thích
    max_pages: int = 1,
    hl: str = "vi",
    gl: str = "vn"
) -> pd.DataFrame:
    ll = f"@{lat},{lng},15z"
    params = {
        "engine": "google_maps",
        "type": "search",
        "q": keyword if keyword else "nhà hàng",
        "ll": ll,
        "hl": hl,       # ngôn ngữ giao diện
        "gl": gl,       # quốc gia
        "api_key": api_key,
    }

    all_results = []
    page = 0
    next_token = None

    while page < max_pages:
        if next_token:
            params["next_page_token"] = next_token

        resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
        # HTTP lỗi -> raise luôn
        resp.raise_for_status()
        data = resp.json()

        # SerpApi hết quota/invalid key vẫn trả HTTP 200 kèm "error" trong JSON
        if "error" in data:
            raise RuntimeError(f"SerpApi error: {data['error']}")

        batch = _extract_results(data)
        all_results.extend(batch)

        # phân trang nếu có
        pagination = data.get("serpapi_pagination") or {}
        next_token = pagination.get("next_page_token")
        page += 1
        if not next_token:
            break

    rows = []
    for p in all_results:
        gps = p.get("gps_coordinates", {})
        price_symbol = p.get("price")
        price_rng = _symbol_to_range(price_symbol)

        price_min = price_max = price_avg = None
        if price_rng:
            price_min, price_max = price_rng
            price_avg = int((price_min + price_max)//2)

        hours = p.get("hours")
        open_now = hours.get("open_now") if isinstance(hours, dict) else None

        rows.append({
            "place_id": p.get("place_id") or p.get("data_id") or p.get("cid"),
            "name": p.get("title"),
            "lat": gps.get("latitude"),
            "lng": gps.get("longitude"),
            "price_min": price_min,
            "price_max": price_max,
            "price_avg": price_avg,
            "rating_avg": p.get("rating"),
            "rating_count": p.get("reviews"),
            "open_hours": hours,
            "open_now": open_now,
            "tags": [p.get("type")] if p.get("type") else [],
            "category": p.get("type"),
            "ward": None,
            "address": p.get("address"),
            "price_level": price_symbol,
            "website": p.get("website"),
            "phone": p.get("phone"),
            "google_url": p.get("link"),
            "types": [p.get("type")] if p.get("type") else [],
        })

    df = pd.DataFrame(rows)

    # Nếu rỗng, báo lỗi rõ ràng để UI Streamlit hiển thị
    if df.empty:
        raise RuntimeError("SerpApi trả về 0 kết quả. Hãy thử: kiểm tra API key, đổi keyword (vd: 'restaurant'), hoặc tăng zoom/đổi vị trí.")

    order = ["place_id","name","lat","lng","price_min","price_max","price_avg",
             "rating_avg","rating_count","open_hours","open_now","tags","category",
             "ward","address","price_level","website","phone","google_url","types"]
    final_cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    return df[final_cols]
