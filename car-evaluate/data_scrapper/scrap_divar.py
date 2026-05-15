import csv
import re
import time
import random
import copy
import json
from urllib.parse import quote
from curl_cffi import requests

SEARCH_API_URL = "https://api.divar.ir/v8/postlist/w/search"
DETAIL_API_URL = "https://api.divar.ir/v8/posts-v2/web/"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9,fa;q=0.8",
    "content-type": "application/json",
    "origin": "https://divar.ir",
    "referer": "https://divar.ir/",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
}

BASE_PAYLOAD = {
    "city_ids": ["1"],
    "source_view": "SEARCH_DESCRIPTOR_ROW",
    "disable_recommendation": False,
    "map_state": {"camera_info": {"bbox": {}}},
    "search_data": {
        "form_data": {
            "data": {
                "category": {"str": {"value": "light"}},
                "brand_model": {"repeated_string": {"value": ["Tara"]}},
            }
        },
        "server_payload": {
            "@type": "type.googleapis.com/widgets.SearchData.ServerPayload",
            "additional_form_data": {
                "data": {"sort": {"str": {"value": "sort_date"}}}
            }
        }
    }
}

def fa_to_en(text):
    if text is None: return None
    text = str(text)
    for fa, en in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789"): text = text.replace(fa, en)
    for ar, en in zip("٠١٢٣٤٥٦٧٨٩", "0123456789"): text = text.replace(ar, en)
    return text

def clean_number(text):
    text = fa_to_en(text)
    if not text: return None
    if "صفر" in text: return 0
    nums = re.sub(r"[^\d]", "", text)
    return int(nums) if nums else None

def make_url(title, token):
    title = re.sub(r"[^\w\s\u0600-\u06FF]", " ", title or "divar-post")
    title = re.sub(r"\s+", "-", title.strip())
    return f"https://divar.ir/v/{quote(title)}/{token}"

def extract_year(text):
    m = re.search(r"(13[8-9]\d|14[0-1]\d|40[0-9])", fa_to_en(text or ""))
    if not m: return None
    year = int(m.group(1))
    return year + 1000 if 400 <= year <= 409 else year

def calc_age(year):
    return 1405 - year if year else None


def crawl_json_for_data(json_obj):
    extracted_fields = {}
    descriptions = []

    def crawl(node):
        if isinstance(node, dict):
            if "title" in node and isinstance(node["title"], str):
                if "value" in node and isinstance(node["value"], str):
                    extracted_fields[node["title"]] = node["value"]
                elif "descriptive_score" in node and isinstance(node["descriptive_score"], str):
                    extracted_fields[node["title"]] = node["descriptive_score"]

            if "description" in node and isinstance(node["description"], str):
                descriptions.append(node["description"])
                
            for v in node.values(): crawl(v)
        elif isinstance(node, list):
            for item in node: crawl(item)

    crawl(json_obj)
    return extracted_fields, " ".join(descriptions)

def parse_api_json(divar_data):
    fields, full_description = crawl_json_for_data(divar_data)

    mileage_text = fields.get("کارکرد", "")
    model_year_text = fields.get("مدل (سال تولید)") or fields.get("مدل", "")
    year = extract_year(model_year_text)
    fuel_type_text = fields.get("نوع سوخت", "")
    
    has_airbag = "دارد" if "ایربگ" in full_description else "نامشخص"
    is_dual_fuel = "دوگانه" if "دوگانه" in full_description or "دوگانه" in fuel_type_text else "بنزینی"
    tire_quality = "نو" if any(w in full_description for w in ["لاستیک نو", "لاستیک ۱۰۰", "لاستیک 100", "لاستیک صفر"]) else "نامشخص"

    return {
        "price_toman": clean_number(fields.get("قیمت پایه") or fields.get("قیمت")),
        "mileage_km": clean_number(mileage_text),
        "year": year,
        "age": calc_age(year),
        "color": fields.get("رنگ"),
        "body_status": fields.get("بدنه"),
        "engine_status": fields.get("موتور"),
        "chassis_status": fields.get("وضعیت شاسی‌ها") or fields.get("شاسی"),
        "gearbox": fields.get("گیربکس"),
        "fuel_type": is_dual_fuel,
        "insurance": clean_number(fields.get("مهلت بیمهٔ شخص ثالث") or fields.get("مهلت بیمه")),
        "brand_model": fields.get("برند و مدل"),
        "airbag": has_airbag,
        "tire_quality": tire_quality
    }

def fetch_detail(session, token):
    url = f"{DETAIL_API_URL}{token}"
    response = session.get(url, headers=HEADERS, timeout=30)
    
    if response.status_code != 200:
        print(f"   ⚠️ API Error {response.status_code} for token {token}")
        return {}
    
    try:
        json_data = response.json()
        return parse_api_json(json_data)
    except json.JSONDecodeError:
        print(f"   ⚠️ Failed to decode JSON for {token}")
        return {}

def make_next_pagination(api_pagination, current_page, cumulative_widgets_count):
    pagination = {
        "@type": "type.googleapis.com/post_list.PaginationData",
        "page": current_page, 
        "layer_page": current_page,
        "cumulative_widgets_count": cumulative_widgets_count,
    }
    
    last_date = api_pagination.get("last_post_date") or api_pagination.get("lastPostDate")
    if last_date: pagination["last_post_date"] = last_date
        
    search_uid = api_pagination.get("search_uid") or api_pagination.get("searchUid")
    if search_uid: pagination["search_uid"] = search_uid
        
    viewed_tokens = api_pagination.get("viewed_tokens") or api_pagination.get("viewedTokens")
    if viewed_tokens: pagination["viewed_tokens"] = viewed_tokens
        
    bookmark_info = api_pagination.get("search_bookmark_info") or api_pagination.get("searchBookmarkInfo")
    if bookmark_info: pagination["search_bookmark_info"] = bookmark_info
        
    first_page = api_pagination.get("first_page_viewed_at") or api_pagination.get("firstPageViewedAt")
    if first_page: pagination["first_page_viewed_at"] = first_page
        
    return pagination

def collect_data(target_count=100, max_pages=100):
    payload = copy.deepcopy(BASE_PAYLOAD)
    all_rows = []
    seen_tokens = set()
    cumulative_widgets_count = 0

    session = requests.Session(impersonate="chrome120")
    session.headers.update(HEADERS)

    print("Initializing session and fetching fresh cookies...")
    try:
        session.get("https://divar.ir/", timeout=30)
        time.sleep(2)
    except Exception as e:
        print(f"Failed to get initial cookies: {e}")

    for page in range(1, max_pages + 1):
        print(f"\n--- Search page {page} ---")

        response = session.post(SEARCH_API_URL, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"Search error {response.status_code}")
            break

        js = response.json()
        widgets = js.get("list_widgets", [])
        posts = [w for w in widgets if w.get("widget_type") == "POST_ROW"]

        if not posts:
            print("No posts found. Stop.")
            break

        for widget in posts:
            data = widget.get("data", {})
            token = data.get("action", {}).get("payload", {}).get("token") or data.get("token")
            title = data.get("title", "")
            
            if not token or token in seen_tokens:
                continue

            seen_tokens.add(token)
            url = make_url(title, token)
            print(f"[{len(all_rows) + 1}/{target_count}] Fetching API: {token}")

            row_data = {"token": token, "url": url, "title": title}
            
            detail = fetch_detail(session, token)
            row_data.update(detail)
            
            all_rows.append(row_data)

            if len(all_rows) >= target_count:
                return all_rows

            delay = random.uniform(8.0, 14.0)
            print(f"   Waiting {delay:.1f}s...")
            time.sleep(delay)

        cumulative_widgets_count += len(posts)
        
        api_pagination = js.get("pagination") or js.get("pagination_data") or {}
        if not api_pagination: break

        page_data = api_pagination.get("data") if "data" in api_pagination else api_pagination

        payload["pagination_data"] = make_next_pagination(
            page_data, 
            current_page=page, 
            cumulative_widgets_count=cumulative_widgets_count
        )

        time.sleep(random.uniform(3.0, 5.0))

    return all_rows

def save_csv(rows, filename="divar_tara_ml_dataset.csv"):
    fields = [
        "title", "token", "url", "price_toman", "mileage_km",
        "year", "age", "color", "body_status", "engine_status", 
        "chassis_status", "gearbox", "fuel_type", "insurance", "brand_model",
        "airbag", "tire_quality"
    ]

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {filename}")

if __name__ == "__main__":
    print("Starting Divar Scraper for ML Homework Dataset...")
    rows = collect_data(target_count=150, max_pages=100)
    save_csv(rows)