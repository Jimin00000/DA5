# ================================================================
# 자사몰+스마트스토어 vs 쿠팡 경쟁사 - 소구점 비교
# (CJ온스타일은 텍스트 리뷰가 없어서 제외)
#
# 사전 조건:
#   - df: crema_reviews_full.csv 원본 (product_name, review_text, channel 등)
#   - long_df_self: analyze_reviews_with_meta(df)의 결과에 competitor_category가
#     추가된 long-format (문장 단위, category/sentiment 포함)
#   - sentiment_df: 쿠팡/CJ 노트북에서 만든 경쟁사 감성분석 결과
#     (제품, 카테고리, 긍정, 부정, 중립 컬럼)
#   - PRODUCT_CATEGORY_MAP: 쿠팡/CJ 노트북에서 쓰던 제품->카테고리 매핑
#
# appeal_keywords의 카테고리명(맛/복용편의, 효능/효과 등)과
# 경쟁사 분석의 카테고리명(맛/복용편의, 효능/효과 등)이 동일하므로
# 그대로 합쳐서 비교 가능
# ================================================================
import pandas as pd
from IPython.display import display

from map_to_competitor_categories import expand_to_categories, COMPETITOR_CATEGORIES


def build_self_brand_long_df(df: pd.DataFrame, analyze_reviews_with_meta) -> pd.DataFrame:
    """
    crema 원본 df -> competitor_category가 붙은 문장 단위 long_df 생성.
    (세트 상품은 두 카테고리 모두에 중복 포함됨)
    """
    df_expanded = expand_to_categories(df)
    long_df = analyze_reviews_with_meta(df_expanded)
    # competitor_category를 review_id 기준으로 다시 합쳐줌
    cat_map = df_expanded[["review_id", "competitor_category"]]
    long_df = long_df.merge(cat_map, on="review_id", how="left")
    return long_df


def self_brand_summary(long_df_self: pd.DataFrame) -> pd.DataFrame:
    """자사(자사몰+스마트스토어 통합) 제품군 x 소구점 카테고리 긍정/부정 건수"""
    pos = (
        long_df_self[long_df_self["sentiment"] == "positive"]
        .groupby(["competitor_category", "category"]).size()
        .unstack(fill_value=0)
    )
    neg = (
        long_df_self[long_df_self["sentiment"] == "negative"]
        .groupby(["competitor_category", "category"]).size()
        .unstack(fill_value=0)
    )
    return pos, neg


def competitor_summary(sentiment_df: pd.DataFrame, product_category_map: dict) -> pd.DataFrame:
    """
    쿠팡 경쟁사 sentiment_df(제품,카테고리,긍정,부정,중립)를
    제품 -> 제품군으로 합산
    """
    df = sentiment_df.copy()
    df["제품군"] = df["제품"].map(product_category_map)
    df = df[df["제품군"].notna()]

    pos = df.pivot_table(index="제품군", columns="카테고리", values="긍정", aggfunc="sum", fill_value=0)
    neg = df.pivot_table(index="제품군", columns="카테고리", values="부정", aggfunc="sum", fill_value=0)
    return pos, neg


def build_side_by_side(self_pos, self_neg, comp_pos, comp_neg, normalize: bool = True,
                        min_sample_size: int = 10):
    """
    자사 vs 경쟁사를 같은 카테고리 축으로 나란히 비교.
    normalize=True면 각 그룹(자사/경쟁사) 내부에서 비율(%)로 변환해서
    표본 크기 차이(자사 리뷰 수 vs 경쟁사 리뷰 수)를 보정함.

    min_sample_size: 분모(긍정+부정 합계)가 이 값 미만이면 표본부족으로 표시.
                      셀 단위 비율은 신뢰하기 어려우므로 보고서에 그대로 쓰지 말 것.
    """
    def to_pct(pos_df, neg_df):
        total = pos_df.add(neg_df, fill_value=0)
        pos_pct = (pos_df / total.replace(0, pd.NA) * 100).fillna(0).round(1)
        neg_pct = (neg_df / total.replace(0, pd.NA) * 100).fillna(0).round(1)
        return pos_pct, neg_pct

    # 비율 변환 전에 원본(절대건수)으로 분모를 미리 계산해둠
    self_total_raw = self_pos.add(self_neg, fill_value=0)
    comp_total_raw = comp_pos.add(comp_neg, fill_value=0)

    if normalize:
        self_pos_disp, self_neg_disp = to_pct(self_pos, self_neg)
        comp_pos_disp, comp_neg_disp = to_pct(comp_pos, comp_neg)
    else:
        self_pos_disp, self_neg_disp = self_pos, self_neg
        comp_pos_disp, comp_neg_disp = comp_pos, comp_neg

    def get_val(df, cat, appeal_cat):
        if cat in df.index and appeal_cat in df.columns:
            return df.loc[cat, appeal_cat]
        return 0

    rows = []
    for cat in COMPETITOR_CATEGORIES:
        for appeal_cat in sorted(set(self_pos.columns) | set(comp_pos.columns)):
            self_n = get_val(self_total_raw, cat, appeal_cat)
            comp_n = get_val(comp_total_raw, cat, appeal_cat)

            # 경쟁사 데이터 자체가 없는 경우(투데이D3 등) N/A로 명시
            comp_data_exists = cat in comp_total_raw.index and appeal_cat in comp_total_raw.columns
            comp_pos_val = get_val(comp_pos_disp, cat, appeal_cat)
            comp_neg_val = get_val(comp_neg_disp, cat, appeal_cat)

            # 표본부족 플래그: 분모가 min_sample_size 미만이면 신뢰도 낮음
            reliability = "OK"
            if not comp_data_exists or comp_n == 0:
                reliability = "경쟁사 데이터 없음(N/A)"
                comp_pos_val, comp_neg_val = pd.NA, pd.NA
            elif comp_n < min_sample_size:
                reliability = f"표본부족(n={int(comp_n)})"

            rows.append({
                "제품군": cat, "소구점": appeal_cat,
                "자사_긍정": get_val(self_pos_disp, cat, appeal_cat),
                "자사_부정": get_val(self_neg_disp, cat, appeal_cat),
                "자사_표본수": int(self_n),
                "경쟁사_긍정": comp_pos_val,
                "경쟁사_부정": comp_neg_val,
                "경쟁사_표본수": int(comp_n),
                "신뢰도": reliability,
            })
    return pd.DataFrame(rows).set_index(["제품군", "소구점"])


def display_comparison(compare_df: pd.DataFrame, is_pct: bool = True):
    """
    compare_df: build_side_by_side() 결과 (자사_긍정/부정/표본수, 경쟁사_긍정/부정/표본수, 신뢰도 포함)
    신뢰도가 'OK'가 아닌 행(표본부족/데이터없음)은 노란/회색으로 강조해서
    숫자만 보고 과신하지 않도록 표시함.
    """
    pct_cols = ["자사_긍정", "자사_부정", "경쟁사_긍정", "경쟁사_부정"]
    int_cols = ["자사_표본수", "경쟁사_표본수"]

    def row_style(row):
        idx = compare_df.index.get_loc(row.name)
        base_bg = "#FFFFFF" if idx % 2 == 0 else "#F2F2F2"
        reliability = row.get("신뢰도", "OK")
        if reliability == "경쟁사 데이터 없음(N/A)":
            bg = "#EFEFEF"
        elif isinstance(reliability, str) and reliability.startswith("표본부족"):
            bg = "#FFF3CD"  # 옅은 노란색 - 주의 필요
        else:
            bg = base_bg
        return [f"background-color: {bg}; color: #111111;"] * len(row)

    fmt_dict = {col: "{:.1f}%" for col in pct_cols} if is_pct else {col: "{:.0f}" for col in pct_cols}
    fmt_dict.update({col: "{:.0f}" for col in int_cols})

    styled = (
        compare_df.style
        .apply(row_style, axis=1)
        .format(fmt_dict, na_rep="N/A")
        .set_properties(**{
            "font-size": "12px", "font-family": "Arial, sans-serif",
            "text-align": "center", "border": "1px solid #d0d0d0", "padding": "6px 10px",
        })
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#2F5496"), ("color", "white"),
                ("font-weight", "bold"), ("font-size", "11px"),
                ("text-align", "center"), ("padding", "7px"),
            ]},
            {"selector": "th.row_heading", "props": [
                ("background-color", "#E8EEF7"), ("color", "#1F3864"),
                ("text-align", "left"), ("font-weight", "bold"),
                ("font-size", "11px"), ("padding", "6px 10px"),
            ]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
            {"selector": "caption", "props": [
                ("caption-side", "top"), ("font-size", "14px"), ("font-weight", "bold"),
                ("color", "#1F3864"), ("padding", "10px 0 6px 0"), ("text-align", "left"),
            ]},
        ])
        .set_caption(
            "자사몰+스마트스토어 vs 경쟁사(쿠팡) 소구점 비교 "
            "(노란색=경쟁사 표본 10건 미만 — 참고용, 회색=경쟁사 데이터 없음)"
        )
    )
    display(styled)
