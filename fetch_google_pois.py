# fetch_google_pois.py
# Helper fetcher for Google Places (Nearby Search + optional Place Details)
# Usage:
#   from fetch_google_pois import get_google_pois
#   df = get_google_pois((10.7798,106.6992), 3000, "restaurant", API_KEY, cache_csv="hcm_pois_google.csv")

import time
import requests
import pandas as pd
from typing import Tuple, Optional, Dict, Any

GOOGLE_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

def _nearby_search(api_key: str, location: Tuple[float, float], radius_m: int, keyword: str,
                   pagetoken: Optional[str] = None) -> Dict[str, Any]:
    params = {
        "key": api_key,
        "location": f"{location[0]},{location[1]}",
        "radius": radius_m,
        "keyword": keyword,
    }
    if pagetoken:
        params["pagetoken"] = pagetoken
    resp = requests.get(GOOGLE_NEARBY_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _place_details(api_key: str, place_id: str,
                   fields: str = "opening_hours,price_level,website,formatted_phone_number,url") -> Dict[str, Any]:
    params = {"key": api_key, "place_id": place_id, "fields": fields}
    resp = requests.get(GOOGLE_DETAILS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def _normalize_result(r: Dict[str, Any]) -> Dict[str, Any]:
    loc = r.get("geometry", {}).get("location", {})
    return {
        "place_id": r.get("place_id"),
        "name": r.get("name"),
        "lat": loc.get("lat"),
        "lng": loc.get("lng"),
        "rating_avg": r.get("rating", 0.0),
        "rating_count": r.get("user_ratings_total", 0),
        "address": r.get("vicinity", ""),
        "price_level": r.get("price_level", None),
        "types": r.get("types", []),

        # Fields to align with our Smart Tourism schema
        "price_min": None, "price_max": None, "price_avg": None,
        "tags": r.get("types", []),
        "category": "Restaurant" if "restaurant" in r.get("types", []) else (r.get("types", [None])[0] if r.get("types") else None),
        "ward": None,
        "open_now": r.get("opening_hours", {}).get("open_now") if r.get("opening_hours") else None,
        "open_hours": None,
    }

def _merge_details(base: Dict[str, Any], details: Dict[str, Any]) -> Dict[str, Any]:
    if details.get("status") != "OK":
        return base
    result = details.get("result", {})
    opening_hours_raw = result.get("opening_hours", {})
    periods = opening_hours_raw.get("periods")
    weekday_text = opening_hours_raw.get("weekday_text")
    open_hours = None
    if periods:
        day_map = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        open_hours = {d: [] for d in day_map}
        for p in periods:
            o, c = p.get("open"), p.get("close")
            if o and c:
                try:
                    d_label = day_map[o.get("day", 0) % 7]
                    ot, ct = o.get("time", "0000"), c.get("time", "2359")
                    ot_fmt, ct_fmt = f"{ot[:2]}:{ot[2:]}", f"{ct[:2]}:{ct[2:]}"
                    open_hours[d_label].append([ot_fmt, ct_fmt])
                except Exception:
                    pass
    elif weekday_text:
        # Fallback to human text if periods aren't available
        open_hours = {"text": weekday_text}

    merged = {**base}
    merged.update({
        "price_level": result.get("price_level", merged.get("price_level")),
        "website": result.get("website"),
        "phone": result.get("formatted_phone_number"),
        "google_url": result.get("url"),
        "open_hours": open_hours if open_hours is not None else merged.get("open_hours"),
    })
    return merged

def get_google_pois(location: Tuple[float, float],
                    radius_m: int,
                    keyword: str,
                    api_key: str,
                    max_pages: int = 3,
                    fetch_details: bool = True,
                    details_fields: str = "opening_hours,price_level,website,formatted_phone_number,url",
                    sleep_between_pages: float = 2.0,
                    sleep_between_details: float = 0.1,
                    cache_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch POIs from Google Places and return a DataFrame compatible with our recommender.
    - Nearby Search pagination handled with next_page_token (sleep is required by Google).
    - Optional Place Details to enrich opening_hours/website/phone/url/price_level.
    - Optional CSV caching.
    """
    all_raw, page, next_page_token = [], 0, None
    while page < max_pages:
        data = _nearby_search(api_key, location, radius_m, keyword, next_page_token)
        status = data.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            raise RuntimeError(f"Nearby Search status={status}, error={data.get('error_message')}")
        all_raw.extend(data.get("results", []))
        next_page_token = data.get("next_page_token")
        page += 1
        if not next_page_token:
            break
        time.sleep(sleep_between_pages)  # Google recommends ~2s before using next_page_token

    # normalize & deduplicate by place_id
    uniq, out = {}, []
    for r in all_raw:
        rec = _normalize_result(r)
        pid = rec.get("place_id")
        if pid and pid not in uniq:
            uniq[pid] = rec
            out.append(rec)

    # enrich via Place Details
    if fetch_details:
        for rec in out:
            pid = rec.get("place_id")
            if not pid:
                continue
            try:
                det = _place_details(api_key, pid, fields=details_fields)
                rec.update(_merge_details(rec, det))
            except Exception:
                # be robust to partial failures / quota errors
                pass
            time.sleep(sleep_between_details)

    df = pd.DataFrame(out)

    # Heuristic mapping price_level (0–4) -> VND range for our scoring
    price_map = {0:(20000,70000), 1:(40000,120000), 2:(80000,200000), 3:(150000,350000), 4:(300000,700000)}
    if not df.empty:
        def _apply_price(row):
            lvl = row.get("price_level")
            rng = price_map.get(lvl) if pd.notna(lvl) else None
            if rng:
                row["price_min"], row["price_max"] = rng
                row["price_avg"] = int((rng[0]+rng[1])//2)
            return row
        df = df.apply(_apply_price, axis=1)

    if cache_csv:
        try:
            df.to_csv(cache_csv, index=False)
        except Exception:
            pass

    # Order columns to match our recommender
    order = [
        "place_id","name","lat","lng",
        "price_min","price_max","price_avg",
        "rating_avg","rating_count",
        "open_hours","open_now",
        "tags","category","ward",
        "address","price_level","website","phone","google_url","types"
    ]
    final_cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    return df[final_cols]
