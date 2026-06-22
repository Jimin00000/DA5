#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파이토뉴트리 공식몰 - 크리마(Crema) 리뷰 위젯 API 크롤러

배경
- bdId=goodsreview 게시판은 JS 동적 로딩이라 정적 요청으로 데이터를 못 가져왔음
- 실제 리뷰 데이터는 크리마(cre.ma)라는 외부 리뷰 위젯 서비스가 제공하는
  JSON API에서 내려옴: https://review6.cre.ma/api/phytonutri.kr/reviews
- 이 API를 직접 호출하면 HTML 파싱 없이 안정적으로 리뷰를 받을 수 있음

이 API가 자사몰(고도몰) 직접 작성 리뷰와 네이버 스마트스토어 리뷰를
모두 한 곳에 모아서 보여주므로, review_vendor 필드로 두 채널을 구분해서
한번에 수집/분석할 수 있음 (사용자가 원했던 정확히 그 형태)

채널 구분 (실측 확인됨)
- review_vendor == "smart_store"  -> 네이버 스마트스토어 리뷰
- review_vendor is None           -> 자사몰(고도몰) 직접 작성 리뷰

먼저 1~2페이지로 테스트하세요:
    df_test, info = crawl_crema_reviews(end_page=2)
    print(info)
    df_test.head(10)

주의
- secure_device_token이 세션/기기에 종속될 수 있어 일정 시간 후 만료되거나
  바뀔 수 있습니다. 만약 403/빈 응답이 나오면, 브라우저 Network 탭에서
  reviews 요청을 다시 열어 secure_device_token 값을 최신으로 교체하세요.
- category_id=1로 확인했으나 이게 '전체' 카테고리인지 특정 상품 카테고리인지는
  미확인 상태입니다. 처음 테스트 시 다양한 product_name이 섞여 나오면
  전체 카테고리로 보아도 무방합니다.
"""
from __future__ import annotations

import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

API_URL = "https://review6.cre.ma/api/phytonutri.kr/reviews"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "referer": "https://review6.cre.ma/v2/phytonutri.kr/mobile/reviews/list_v3",
    "user-agent": (
        "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Mobile Safari/537.36"
    ),
}

# 브라우저에서 직접 복사한 토큰. 만료되면 이 값만 교체하면 됩니다.
DEFAULT_SECURE_DEVICE_TOKEN = (
    "V273152637667ecc4fb28164c75023ee707e429e113eb2f5d9006f031a39926bc26b009c61ec2344e8566e03c9e74feea3"
)


def _parse_created_at(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # 예: '2026-06-11T23:04:15+09:00'
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _classify_vendor(review_vendor: Optional[str]) -> str:
    if review_vendor == "smart_store":
        return "스마트스토어"
    if review_vendor is None:
        return "자사몰(고도몰)"
    return f"기타({review_vendor})"


def fetch_review_page(
    session: requests.Session,
    page: int,
    category_id: int = 1,
    secure_device_token: str = DEFAULT_SECURE_DEVICE_TOKEN,
    timeout: int = 20,
) -> Tuple[Optional[dict], str, int]:
    params = {
        "secure_device_token": secure_device_token,
        "fields": "has_media,reviews.evaluation_properties,reviews.ai_summary,"
                  "reviews.with_parent_reviews,reviews.customer_properties",
        "iframe_id": "crema-reviews-1",
        "widget_id": 33,
        "widget_style": "list_v3",
        "locale": "ko-KR",
        "app": 0,
        "device": "mobile",
        "page": page,
        "category_id": category_id,
        "iframe": 1,
    }
    try:
        r = session.get(API_URL, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json(), "ok", r.status_code
        return None, f"http_{r.status_code}", r.status_code
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", 0


def flatten_review(review: dict, page: int) -> dict:
    created_dt = _parse_created_at(review.get("created_at"))
    images = review.get("images") or []
    return {
        "review_id": review.get("id"),
        "page": page,
        "score": review.get("score"),
        "review_vendor_raw": review.get("review_vendor"),
        "channel": _classify_vendor(review.get("review_vendor")),
        "review_text": review.get("filtered_message", ""),
        "created_at": review.get("created_at"),
        "created_date": created_dt.date().isoformat() if created_dt else None,
        "author_display_name": review.get("user_display_name"),
        "author_grade": review.get("author_grade"),
        "has_media": bool(review.get("media_count")),
        "media_count": review.get("media_count", 0),
        "image_count": len(images),
        "image_urls": "|".join(img.get("url", "") for img in images) if images else "",
        "product_code": review.get("product_code"),
        "product_name": review.get("product_name"),
        "product_url": review.get("product_url"),
        "product_meta_score": review.get("product_meta_score"),
        "product_meta_reviews_count": review.get("product_meta_reviews_count"),
        "likes_count": review.get("likes_count"),
        "comments_count": review.get("comments_count"),
        "ai_summary": review.get("ai_summary"),
        "product_options": review.get("product_options"),
        "customer_properties": review.get("customer_properties"),
    }


def crawl_crema_reviews(
    since_date: Optional[str] = None,    # 예: "2025-01-01". None이면 날짜 제한 없음
    category_id: int = 1,
    start_page: int = 1,
    end_page: Optional[int] = None,      # 안전장치. None이면 데이터 끝까지(또는 since_date까지) 진행
    secure_device_token: str = DEFAULT_SECURE_DEVICE_TOKEN,
    delay: float = 0.5,
    jitter: float = 0.3,
    outdir: str = "phytonutri_crema_reviews",
    checkpoint_every: int = 20,
    stop_after_n_consecutive_older: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    크리마 리뷰 API를 페이지 단위로 호출해서 전체(또는 since_date까지) 수집.
    review_vendor로 자사몰/스마트스토어를 한 번에 구분해서 담음.
    """
    since_dt = datetime.strptime(since_date, "%Y-%m-%d") if since_date else None
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    all_rows: List[dict] = []
    status_records: List[dict] = []
    consecutive_older_pages = 0
    stopped_reason = None
    page = start_page

    while True:
        if end_page and page > end_page:
            stopped_reason = f"end_page({end_page}) 도달"
            break

        data, status, code = fetch_review_page(
            session, page, category_id=category_id, secure_device_token=secure_device_token
        )
        rows_count = 0
        if data:
            reviews = data.get("reviews", [])
            rows_count = len(reviews)
            for rv in reviews:
                all_rows.append(flatten_review(rv, page))

            # 다음 페이지 존재 여부 (pagy.next가 없으면 마지막 페이지)
            pagy = data.get("pagy", {})
            has_next = pagy.get("next") is not None
        else:
            has_next = False

        status_records.append({
            "page": page, "status": status, "http_code": code, "rows": rows_count,
        })
        print(f"page={page} status={status} rows={rows_count}")

        # ── 날짜 기준 종료 판정 ──────────────────────────────────
        if since_dt and rows_count > 0:
            page_dates = [
                _parse_created_at(r.get("created_at")) for r in data.get("reviews", [])
            ]
            page_dates = [d for d in page_dates if d is not None]
            page_dates_naive = [d.replace(tzinfo=None) for d in page_dates]
            if page_dates_naive and all(d < since_dt for d in page_dates_naive):
                consecutive_older_pages += 1
            else:
                consecutive_older_pages = 0

        if page % checkpoint_every == 0 and all_rows:
            pd.DataFrame(all_rows).to_csv(
                outdir_p / "crema_reviews_checkpoint.csv", index=False, encoding="utf-8-sig"
            )
            print(f"  [checkpoint] page={page} 누적 rows={len(all_rows)}")

        if not rows_count and not has_next:
            stopped_reason = "API가 빈 응답을 반환 (데이터 끝 또는 토큰 만료 의심)"
            break
        if not has_next:
            stopped_reason = "pagy.next 없음 (마지막 페이지 도달)"
            break
        if since_dt and consecutive_older_pages >= stop_after_n_consecutive_older:
            stopped_reason = (
                f"'{since_date}'보다 오래된 리뷰가 {stop_after_n_consecutive_older}페이지 "
                f"연속 발견되어 중단"
            )
            break

        page += 1
        time.sleep(max(0, delay) + random.uniform(0, jitter))

    df = pd.DataFrame(all_rows)
    final_path = outdir_p / "crema_reviews_full.csv"
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(status_records).to_csv(outdir_p / "crawl_page_status.csv", index=False, encoding="utf-8-sig")

    if since_dt and not df.empty:
        df["_created_dt"] = df["created_at"].apply(_parse_created_at)
        df["_created_dt_naive"] = df["_created_dt"].apply(lambda d: d.replace(tzinfo=None) if d else None)
        df_filtered = df[df["_created_dt_naive"].isna() | (df["_created_dt_naive"] >= since_dt)].copy()
        df_filtered = df_filtered.drop(columns=["_created_dt", "_created_dt_naive"])
        filtered_path = outdir_p / f"crema_reviews_since_{since_date}.csv"
        df_filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")
    else:
        df_filtered = df
        filtered_path = final_path

    channel_counts = df["channel"].value_counts().to_dict() if not df.empty else {}

    summary = {
        "stopped_reason": stopped_reason,
        "pages_crawled": page - start_page + 1,
        "rows_total": len(df),
        "rows_after_date_filter": len(df_filtered),
        "channel_breakdown_before_filter": channel_counts,
        "saved_full_csv": str(final_path),
        "saved_filtered_csv": str(filtered_path),
    }
    print(summary)
    return df_filtered, summary
