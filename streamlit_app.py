import streamlit as st
import pandas as pd
import datetime
from math import radians, sin, cos, asin, sqrt

# Optional providers
GOOGLE_AVAILABLE = False
SERP_AVAILABLE = False
try:
    from fetch_google_pois import get_google_pois
    GOOGLE_AVAILABLE = True
except Exception:
    pass

try:
    from fetch_serpapi_pois import get_serpapi_pois
    SERP_AVAILABLE = True
except Exception:
    pass

# ---------------- Utilities ----------------

def as_int_or_none(x):
    import pandas as pd
    try:
        return None if x is None or (isinstance(x, float) and pd.isna(x)) else int(x)
    except Exception:
        return None

def as_float_or_zero(x):
    import pandas as pd
    try:
        return 0.0 if x is None or (isinstance(x, float) and pd.isna(x)) else float(x)
    except Exception:
        return 0.0


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def weekday_name(dt: datetime.datetime):
    return ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dt.weekday()]

def parse_hhmm(s: str):
    hh, mm = s.split(":")
    return int(hh), int(mm)

def within(time_str, start_str, end_str):
    sh, sm = parse_hhmm(start_str)
    eh, em = parse_hhmm(end_str)
    th, tm = parse_hhmm(time_str)
    start_min = sh*60 + sm
    end_min = eh*60 + em
    t_min = th*60 + tm
    if start_min <= end_min:
        return start_min <= t_min <= end_min
    return t_min >= start_min or t_min <= end_min
"""
def is_open(row, dt: datetime.datetime):
    day = weekday_name(dt)
    oh = row.get("open_hours")
    if isinstance(oh, dict) and "text" not in oh and day in oh:
        slots = oh.get(day, [])
        t = dt.strftime("%H:%M")
        for s, e in slots:
            if within(t, s, e):
                return True
        return False
    if "open_now" in row:
        return bool(row.get("open_now", True))
    return True
"""

def is_open(row, dt):
    # Ưu tiên thông tin open_now nếu có
    if "open_now" in row:
        val = row.get("open_now")
        if val is True:
            return True
        if val is False:
            return False
        # val is None -> đừng loại, cho qua
        return True

    # Nếu có cấu trúc periods chuẩn (Mon/Tue/...), kiểm tra kỹ
    oh = row.get("open_hours")
    if isinstance(oh, dict) and "text" not in oh:
        day = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dt.weekday()]
        slots = oh.get(day)
        if isinstance(slots, list) and slots:
            t = dt.strftime("%H:%M")

            def parse_hhmm(s):
                hh, mm = s.split(":")
                return int(hh), int(mm)

            def within(time_str, start_str, end_str):
                sh, sm = parse_hhmm(start_str)
                eh, em = parse_hhmm(end_str)
                th, tm = parse_hhmm(time_str)
                start_min = sh*60 + sm
                end_min = eh*60 + em
                t_min = th*60 + tm
                if start_min <= end_min:
                    return start_min <= t_min <= end_min
                return t_min >= start_min or t_min <= end_min

            for s, e in slots:
                if within(t, s, e):
                    return True
            return False

    # Không có thông tin gì chắc chắn -> KHÔNG lọc (cho qua)
    return True


def budget_overlap(user_min, user_max, poi_min, poi_max):
    if pd.isna(poi_min) and pd.isna(poi_max):
        return True
    poi_min = poi_min if not pd.isna(poi_min) else user_min
    poi_max = poi_max if not pd.isna(poi_max) else user_max
    return not (user_max < poi_min or poi_max < user_min)

def price_delta(user_min, user_max, poi_avg):
    if pd.isna(poi_avg):
        return 0.0
    if user_min <= poi_avg <= user_max:
        return 0.0
    if poi_avg < user_min:
        return float(user_min - poi_avg)
    return float(poi_avg - user_max)

def jaccard(a: set, b: set):
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a.union(b))
    return inter/uni if uni > 0 else 0.0

def explain_row(row, tags_user, budget_range, user_loc, dist):
    reasons = []
    row_tags = set(row["tags"]) if isinstance(row.get("tags"), (list, set, tuple)) else set()
    if jaccard(tags_user, row_tags) >= 0.5 and len(row_tags) > 0:
        reasons.append("khớp khẩu vị")
    if budget_overlap(budget_range[0], budget_range[1], row.get("price_min"), row.get("price_max")):
        reasons.append("phù hợp ngân sách")
    if row.get("rating_avg", 0) >= 4.3:
        reasons.append("đánh giá cao")
    if dist <= 1.0:
        reasons.append("rất gần")
    return ", ".join(reasons)

def recommend(df, user_location=(10.7769,106.7009), radius_km=3.0,
              budget_range=(50000, 150000), tags_user=("cay","gia đình"),
              when=None, K=5, w_dist=0.35, w_rating=0.35, w_tag=0.20, w_price=0.10):
    if when is None:
        when = datetime.datetime.now()

    for col in ["lat","lng","rating_avg","price_min","price_max","price_avg","tags","open_hours","open_now","category","ward","name"]:
        if col not in df.columns:
            df[col] = None

    filtered = []
    for _, row in df.iterrows():
        try:
            d = haversine_km(user_location[0], user_location[1], float(row["lat"]), float(row["lng"]))
        except Exception:
            continue
        if d > radius_km:
            continue
        row_dict = row.to_dict()
        if not is_open(row_dict, when):
            continue
        if not budget_overlap(budget_range[0], budget_range[1], row_dict.get("price_min"), row_dict.get("price_max")):
            continue
        filtered.append((row_dict, d))

    if not filtered:
        return pd.DataFrame(columns=["name","ward","category","distance_km","price_avg","rating_avg","tags","score","why"])

    dists = [d for _, d in filtered]
    ratings = [r.get("rating_avg", 0) or 0 for r, _ in filtered]
    pdeltas = [price_delta(budget_range[0], budget_range[1], r.get("price_avg")) for r, _ in filtered]
    dmax = max(dists) if dists else 1.0
    rmin, rmax = (min(ratings), max(ratings)) if ratings else (0.0, 5.0)
    pdmax = max(pdeltas) if pdeltas else 1.0

    scored_rows = []
    tags_user_set = set(tags_user)
    for row, d in filtered:
        distance_score = 1 - (d/dmax if dmax>0 else 1.0)
        rating_score = ((row.get("rating_avg", 0) or 0) - rmin)/((rmax - rmin) if (rmax - rmin) > 0 else 1.0)
        tag_score = jaccard(tags_user_set, set(row.get("tags") or []))
        pdelta = price_delta(budget_range[0], budget_range[1], row.get("price_avg"))
        price_score = 1 - (pdelta/pdmax if pdmax>0 else 0.0)
        S = w_dist*distance_score + w_rating*rating_score + w_tag*tag_score + w_price*price_score

        scored_rows.append({
            "name": row.get("name"),
            "ward": row.get("ward"),
            "category": row.get("category"),
            "distance_km": round(d, 2),
            #"price_avg": int(row.get("price_avg")) if row.get("price_avg") else None,
           #"rating_avg": float(row.get("rating_avg") or 0),
            "price_avg": as_int_or_none(row.get("price_avg")),          # ⬅️ SỬA Ở ĐÂY
            "rating_avg": as_float_or_zero(row.get("rating_avg")), 
            "tags": ", ".join(row.get("tags") or []),
            "score": round(S, 4),
            "why": explain_row(row, tags_user_set, budget_range, user_location, d)
        })

    scored_rows.sort(key=lambda x: x["score"], reverse=True)
    cat_used, ward_used, result = {}, {}, []
    for r in scored_rows:
        cat_ok = cat_used.get(r["category"], 0) < 2 if r["category"] else True
        ward_ok = ward_used.get(r["ward"], 0) < 2 if r["ward"] else True
        if cat_ok and ward_ok:
            result.append(r)
            if r["category"]: cat_used[r["category"]] = cat_used.get(r["category"], 0) + 1
            if r["ward"]: ward_used[r["ward"]] = ward_used.get(r["ward"], 0) + 1
        if len(result) == K:
            break
    if len(result) < K:
        for r in scored_rows:
            if r not in result:
                result.append(r)
            if len(result) == K:
                break
    return pd.DataFrame(result)

# ---------------- UI ----------------
st.set_page_config(page_title="Smart Tourism – Gợi ý Nhà hàng (HCMC)", layout="wide")
st.title("🍜 Gợi ý Nhà hàng – TP.HCM (Smart Tourism Demo)")

col1, col2, col3 = st.columns(3)
with col1:
    lat = st.number_input("Vĩ độ (lat)", value=10.7769, format="%.6f")
    lng = st.number_input("Kinh độ (lng)", value=106.7009, format="%.6f")
    radius_km = st.slider("Bán kính (km)", 0.5, 10.0, 3.0, 0.5)

with col2:
    bmin = st.number_input("Ngân sách tối thiểu (VND)", value=50000, step=10000)
    bmax = st.number_input("Ngân sách tối đa (VND)", value=150000, step=10000)
    K = st.slider("Số gợi ý (K)", 1, 20, 5, 1)

with col3:
    tags_text = st.text_input("Tags (phân tách bởi dấu phẩy)", value="cay, gia đình")
    now = st.checkbox("Dùng thời điểm hiện tại", value=True)
    if not now:
        date = st.date_input("Ngày", value=datetime.date.today())
        hour = st.number_input("Giờ (0-23)", 19, 0, 23)
        minute = st.number_input("Phút (0-59)", 0, 0, 59)
        when = datetime.datetime(date.year, date.month, date.day, hour, minute)
    else:
        when = datetime.datetime.now()

st.divider()

source = st.radio("Chọn nguồn dữ liệu", ["Google Places API", "SerpApi", "CSV mock"],
                  index=0 if GOOGLE_AVAILABLE else (1 if SERP_AVAILABLE else 2))

df = None
if source == "Google Places API":
    st.warning("CHƯA DÙNG ĐƯỢC GOOGLE MAP API VÌ KHÔNG CÓ TIỀN MUA.")
    if not GOOGLE_AVAILABLE:
        st.warning("Chưa có module fetch_google_pois.py — chuyển sang CSV/SerpApi.")
    else:
        api_key = st.text_input("Google API Key", type="password", help="Cần bật Places API + billing")
        if api_key:
            try:
                df = get_google_pois(
                    location=(lat, lng),
                    radius_m=int(radius_km*1000),
                    keyword="restaurant",
                    api_key=api_key,
                    max_pages=2,
                    fetch_details=True,
                    cache_csv=None
                )
                if "tags" in df.columns:
                    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else (x or []))
            except Exception as e:
                st.error(f"Lỗi Google Places: {e}")
    """
    elif source == "SerpApi":
    if not SERP_AVAILABLE:
        st.warning("Chưa có module fetch_serpapi_pois.py — chuyển sang CSV/Google.")
    else:
        serp_key = st.text_input("SerpApi Key", type="password", help="Free plan ~100 requests/tháng")
        if serp_key:
            try:
                df = get_serpapi_pois(lat, lng, "nhà hàng", serp_key, radius_km=radius_km, max_pages=1)
            except Exception as e:
                st.error(f"Lỗi SerpApi: {e}")
    """

elif source == "SerpApi":
    if not SERP_AVAILABLE:
        st.warning("Chưa có module fetch_serpapi_pois.py — chuyển sang CSV/Google.")
    else:
        serp_key = st.text_input("SerpApi Key", type="password", help="Free plan ~100 requests/tháng")
        custom_kw = st.text_input("Từ khoá tìm (mặc định: nhà hàng)", value="nhà hàng")
        if serp_key:
            try:
                df = get_serpapi_pois(lat, lng, custom_kw, serp_key, radius_km=radius_km, max_pages=1)
                df = df.where(pd.notna(df), None)
            except Exception as e:
                st.error(f"Lỗi SerpApi: {e}")
                df = None

    
if df is None:
    try:
        df = pd.read_csv("hcm_pois.csv")
        df["tags"] = df["tags"].apply(lambda x: eval(x) if isinstance(x, str) else x)
        df["open_hours"] = df["open_hours"].apply(lambda x: eval(x) if isinstance(x, str) else x)
        st.info("Đang dùng dữ liệu mock từ hcm_pois.csv (fallback).")
    except Exception:
        st.warning("Không tìm thấy hcm_pois.csv. Vui lòng tải CSV hoặc chọn nguồn khác.")
        df = pd.DataFrame()
        df = df.where(pd.notna(df), None)


tags_user = [t.strip() for t in tags_text.split(",") if t.strip()]
res = recommend(df, (lat,lng), radius_km, (bmin,bmax), tags_user, when, K)
#st.dataframe(res, use_container_width=True)
#st.dataframe(res, width="stretch")
st.write(f"Kết quả gợi ý: {len(res)}")
st.dataframe(res, width="stretch")  # thay use_container_width


st.caption("Nguồn dữ liệu: Google/SerpApi/CSV. Dùng CSV để demo ổn định khi hết quota.")
